from __future__ import annotations
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, Sequence, Tuple, List
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from .trocr_buffer import TROCRBuffer
from app.logger import debug_logger


class PlayerNumberDetector:
    MIN_CONF = 0.60
    IMG_SIZE = (60, 60)

    VERT_FRAC_TOP = 0.25
    VERT_FRAC_BOT = 0.55
    HORZ_MARGIN = 0.20

    def __init__(self, model_dir: str):
        base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed") # nosec
        self.model = PeftModel.from_pretrained(base, model_dir)
        self.processor = TrOCRProcessor.from_pretrained(model_dir) # nosec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    # ---------- crops ----------
    def _crop_dorsal_region(self, frame: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        box_h, box_w = y2 - y1, x2 - x1
        y_top, y_bottom = (
            int(y1 + box_h * self.VERT_FRAC_TOP),
            int(y1 + box_h * self.VERT_FRAC_BOT),
        )
        x_left, x_right = (
            int(x1 + box_w * self.HORZ_MARGIN),
            int(x2 - box_w * self.HORZ_MARGIN),
        )
        if y_bottom <= y_top or x_right <= x_left:
            return np.empty((0, 0), dtype=np.uint8)
        return frame[y_top:y_bottom, x_left:x_right]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bin = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        clean = cv2.morphologyEx(bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        x, y, w, h = cv2.boundingRect(clean)
        crop = clean[y:y + h, x:x + w]
        side = max(h, w)
        square = np.full((side, side), 255, dtype=np.uint8)
        off_y, off_x = (side - h) // 2, (side - w) // 2
        square[off_y: off_y + h, off_x: off_x + w] = crop
        return cv2.resize(square, self.IMG_SIZE)

    def predict_batch(
        self, crops: List[np.ndarray]
    ) -> List[Tuple[Optional[int], float]]:
        if not crops:
            return []
        return self._infer_many(crops)

    def _infer_many(self, crops: List[np.ndarray]) -> List[Tuple[Optional[int], float]]:
        images = [Image.fromarray(c, mode="L").convert("RGB") for c in crops]
        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(self.device)
        debug_logger.debug(f"[PlayerNumberDetector] Input shape: {pixel_values.shape}")

        with torch.no_grad():
            out = self.model.generate(
                pixel_values,
                max_length=10,
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        probs = torch.stack(out.scores, dim=1).softmax(-1)
        debug_logger.debug(f"[PlayerNumberDetector] Texts: {texts}")
        debug_logger.debug(f"[PlayerNumberDetector] Probs: {probs}")

        results = []
        for i, txt in enumerate(texts):
            if not txt.isdigit():
                results.append((None, 0.0))
                continue
            num = int(txt)
            if not (1 <= num <= 99):
                results.append((None, 0.0))
                continue
            tok_ids = out.sequences[i, 1:-1]
            conf = probs[i, torch.arange(tok_ids.shape[0]), tok_ids].mean().item()
            results.append((num, conf))
        debug_logger.debug(f"[PlayerNumberDetector] Results: {results}")
        return results

    def flush_buffer(self, buffer: TROCRBuffer):
        crops, cbks = buffer.flush()
        if not crops:
            return
        preds = self.predict_batch(crops)
        for cb, (num, conf) in zip(cbks, preds):
            debug_logger.debug(f"[PlayerNumberDetector] Callback: {num=}, {conf=}")
            cb(num, conf)
