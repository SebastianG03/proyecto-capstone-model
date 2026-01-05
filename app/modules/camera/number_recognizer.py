# player_number_tesseract.py  (versión unificada)
from __future__ import annotations
import cv2
import numpy as np
import pytesseract
from typing import Optional, Sequence

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class PlayerNumberNP:
    """
    Reconocimiento dorsal 1-99 con Tesseract.
    Entrada: frame (H,W,3) + bbox [x1,y1,x2,y2]
    Salida: int o None
    """

    MIN_CONF = 50
    IMG_SIZE = (60, 60)

    # --- zona de búsqueda (empírico, ajústalo) -----------------
    VERT_FRAC_TOP = 0.25      # inicio del pecho (% altura bbox)
    VERT_FRAC_BOT = 0.55      # fin del pecho
    HORZ_MARGIN = 0.20        # margen lateral para quitar brazos

    def _crop_dorsal_region(
        self,
        frame: np.ndarray,
        bbox: Sequence[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        box_h, box_w = y2 - y1, x2 - x1
        y_top    = int(y1 + box_h * self.VERT_FRAC_TOP)
        y_bottom = int(y1 + box_h * self.VERT_FRAC_BOT)
        x_left   = int(x1 + box_w * self.HORZ_MARGIN)
        x_right  = int(x2 - box_w * self.HORZ_MARGIN)

        if y_bottom <= y_top or x_right <= x_left:
            return np.empty((0, 0), dtype=np.uint8)

        return frame[y_top:y_bottom, x_left:x_right]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Idéntico a tu código anterior."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)

        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        x, y, w, h = cv2.boundingRect(clean)
        crop = clean[y:y + h, x:x + w]

        side = max(h, w)
        square = np.full((side, side), 255, dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = crop

        return cv2.resize(square, self.IMG_SIZE)

    def predict(self,
               frame: np.ndarray,
               bbox) -> Optional[int]:
        """
        frame + bbox -> número 1-99 o None
        """
        dorsal_roi = self._crop_dorsal_region(frame, bbox)
        if dorsal_roi.size == 0:
            return None

        proc = self._preprocess(dorsal_roi)

        txt = pytesseract.image_to_string(
            proc,
            config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        ).strip()

        if not txt.isdigit():
            return None
        num = int(txt)
        if not 1 <= num <= 99:
            return None

        # confianza opcional
        data = pytesseract.image_to_data(
            proc,
            config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
            output_type=pytesseract.Output.DICT
        )
        confs = [int(c) for c in data['conf'] if int(c) > 0]
        avg_conf = np.mean(confs) if confs else 0
        return num if avg_conf >= self.MIN_CONF else None
