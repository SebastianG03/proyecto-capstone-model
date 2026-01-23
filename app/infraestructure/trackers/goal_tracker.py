import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics.models import YOLO

logger = logging.getLogger(__name__)


class GoalTracker:
    """
    Wrapper ligero para un modelo YOLO *dedicado* exclusivamente a
    detectar 'soccer-ball' y 'soccer-goal'.
    """

    def __init__(self,
                 model_path: str | Path,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 device: str = "cuda:0"):
        """
        model_path: ruta al .pt (p.ej. yolov8n-goal-ball.pt)
        device: "cuda:0", "cpu", "mps", ...
        """
        self.model = YOLO(model_path)
        self.model.fuse()  # acelera inferencia
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

        # --- supervision helpers ---
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5)

        logger.info(f"GoalYOLODetector ready: {model_path} on {device}")

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Devuelte sv.Detections con las clases que nos interesan
        ya filtradas por confianza.
        """
        results = self.model(frame,
                             conf=self.conf_thres,
                             iou=self.iou_thres,
                             device=self.device,
                             verbose=False)
        if not results or len(results) == 0:
            return sv.Detections.empty()

        detections = sv.Detections.from_ultralytics(results[0])

        target_names = {"soccer-ball", "soccer-goal"}
        mask = np.isin(detections.data["class_name"], list(target_names))
        return detections[mask]

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return frame
        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(detections.data["class_name"], detections.confidence)
        ]
        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        return annotated