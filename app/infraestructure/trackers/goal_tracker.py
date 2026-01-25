import logging
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from ultralytics.models import YOLO
from app.logger import info_logger


class GoalTracker:
    """
    Wrapper ligero para un modelo YOLO *dedicado* exclusivamente a
    detectar 'soccer-ball' y 'soccer-goal'.
    """

    def __init__(
        self, model_path: str | Path, conf_thres: float = 0.25, iou_thres: float = 0.45
    ):
        """
        model_path: ruta al .pt (p.ej. yolov8n-goal-ball.pt)
        device: "cuda:0", "cpu", "mps", ...
        """

        self.model = YOLO(model_path, task="obb")
        self.model.fuse()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- supervision helpers ---
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5)

        info_logger.info(f"GoalYOLODetector ready: {model_path} on {self.device}")

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Devuelte sv.Detections con las clases que nos interesan
        ya filtradas por confianza.
        """
        results = self.model(
            frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )
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
        annotated = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )
        return annotated
