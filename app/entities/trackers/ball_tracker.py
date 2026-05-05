from asyncio.log import logger
import json
from typing import Any, Dict, List, Optional, Tuple, override
import uuid
import cv2
from cv2.typing import MatLike
from filterpy.kalman import KalmanFilter

import numpy as np
import supervision as sv
from sympy import beta
from ultralytics.models import YOLO

from app.entities.collections.track_collections import TrackCollectionBall
from app.entities.interfaces.tracker_base import Tracker
from sqlalchemy.orm import Session
from app.logger import info_logger, error_logger
import app.entities.utils.global_values_store as value_store            
from app.entities.models.detected_object_data import AnalysisData, DetectedObjectData
import app.entities.utils.tools_context as context
from app.utils import routes

class BallTracker(Tracker):
    def __init__(
        self, model: YOLO, max_age: int = 5, gate: float = 9.21, df: float = 1.0
    ):
        super().__init__(model)
        self.kf = self._create_kf(df)
        self.bbox = []
        self.age = 0
        self.max_age = max_age
        self.gate = gate  # chi2 2-dof 95%
        self.owner_id: Optional[int] = None
        self.possession_time: float = 0.0
        self.last_seen_frame: int = -1
        

    def _create_kf(self, dt: float = 1.0) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.R *= 9  # observacion
        kf.Q[2:, 2:] *= 0.1  # velocidad
        kf.x = np.array([0.0, 0.0, 0.0, 0.0])  # se inicializa en primera deteccion
        kf.P = np.eye(4) * 100.0
        kf.R = np.eye(2) * 9.0
        return kf

    @override
    def reset(self):
        """Resetea el estado interno del tracker."""
        self.kf = self._create_kf()
        self.age = 0
        self.owner_id = None
        self.possession_time = 0.0
        self.last_seen_frame = -1

    @override
    def get_object_tracks(
        self,
        frame: MatLike,
        detections: List,
        cls_names_inv,
        frame_num,
        db: Session,
    ):
        try:
            ball_coordinates, conf = self._extract_ball(detections, cls_names_inv, frame)
            self._update_kf(ball_coordinates, frame_num)
            self._persist_state(frame_num, conf or 0.35)
            logger.debug(f"[BallTracker] frame {frame_num} owner={self.owner_id}")
        except Exception as e:
            logger.exception(
                f"[BallTracker] Error en get_object_tracks frame {frame_num}: {e}"
            )
            raise e
    
    def safe_data(
        self,
        detections: List[np.ndarray],
        frame: MatLike
        ):
        img_path = f"{routes.PLAYER_CUSTOM_DATASET}/{uuid.uuid4()}.jpg"
        cv2.imwrite (img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        label_path = img_path.replace(".jpg", ".txt")
        h, w = frame.shape[:2]
        
        with open(label_path, "w") as f:
            for poly in detections:
                norm_coords = " ".join(
                f"{px/w:.7f} {py/h:.7f}"
                for (px, py) in poly
                )
                f.write(f"{0} {norm_coords}\n")

    def _extract_ball(self, detections: List, cls_names_inv: dict, frame: MatLike):
        try:
            info_logger.info("[BallTracker] extract_ball llamado")
            if detections is None or len(detections) == 0:
                info_logger.info("[BallTracker] No hay detecciones")
                return None, None

            info_logger.info("[BallTracker] extract_ball procesando detecciones")
            self.logger.info(f"[BallTracker] Detecciones recibidas: {detections}")

            bboxes = []
            confidences = []
            polygons = []
            
            for det in detections:
                if det.boxes is not None:
                    for box in det.boxes:
                        info_logger.info(f"Deteccion: {box.xyxy}, clase: {box.cls}, conf: {box.conf}")
                        bbox = box.xyxy.cpu().numpy().squeeze(0)
                        x1, y1, x2, y2 = bbox
                        conf = box.conf.cpu().numpy().squeeze(0)

                        polygons.append(np.array([
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ]))
                        bboxes.append(bbox)
                        confidences.append(conf)            

            if len(bboxes) == 0:
                info_logger.info("[BallTracker] No hay bboxes")
                return self._recover_ball_position()

            self.safe_data(polygons, frame)
            xyxy = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
            confs = np.array(confidences, dtype=np.float32)

            sv_dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=np.zeros(len(bboxes)))
            tracked = self.tracker.update_with_detections(sv_dets)

            xyxy = np.asarray(tracked.xyxy)
            conf = np.asarray(tracked.confidence)

            if not xyxy.any():
                info_logger.info("[BallTracker] No xyxy")
                return self._recover_ball_position()
            
            best_conf = 0
            best_bbox: list[float] = [0.0, 0.0, 0.0, 0.0]
            
            for bbox, conf in zip(xyxy, conf):
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = list(bbox)

            self.bbox = list(map(float, best_bbox))
            cx, cy = self._bbox_to_center(self.bbox)

            return (cx, cy), float(best_conf)
        except Exception as e:
            error_logger.error(f"[BallTracker] Error en extract_ball: {e}")
            raise e

    def _update_kf(
        self, ball_coordinates: Optional[tuple[float, float]], frame_number: int
    ):
        self.kf.predict()
        
        if ball_coordinates is None:
            self.age += 1
            return
        
        z = np.array(ball_coordinates)
        info_logger.info(f"[BallTracker] z={z}, ball_coordinates={ball_coordinates}")

        if np.allclose(self.kf.x, 0):
            self.kf.x = np.array([z[0], z[1], 0.0, 0.0])
            self.age = 0
            self.last_seen_frame = frame_number
            return

        y = z - self.kf.H @ self.kf.x
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        det = np.linalg.det(S)

        if det < 1e-6:
            logger.warning("Matriz S casi singular, saltando actualizacion")
            S_inv = np.linalg.pinv(S)
        else:
            S_inv = np.linalg.inv(S)

        d2 = float((y.T @ S_inv @ y).item())
        if d2 < self.gate:
            self.kf.update(z)
            self.age = 0
            self.last_seen_frame = frame_number
        else:
            logger.debug(f"Deteccion rechazada por gate (d2={d2:.2f})")
            self.age += 1
        

    def get_ball_state(self) -> Optional[Dict[str, Any]]:
        if self.age > self.max_age:
            return None
        logger.debug(f"Recuperando Kalman {self.kf.x} con longitud {len(self.kf.x)}")
        x, y, _, _ = self.kf.x
        return {
            "x": float(x),
            "y": float(y),
            "vx": float(self.kf.x[2]),
            "vy": float(self.kf.x[3]),
            "bbox": self.bbox,
            "owner_id": self.owner_id,
            "possession_time": self.possession_time,
            "last_seen_frame": self.last_seen_frame,
        }

    def _recover_ball_position(self) -> tuple[Optional[tuple[float, float]], Optional[float]]:
        """
        Recupera la posicion del balon usando heuristicas cuando no es detectado.
        """
        info_logger.info("[BallTracker] Aplicando Kalman para recuperar el balon...")

        if self.age < self.max_age and not np.allclose(self.kf.x[:2], 0):
            predicted_x, predicted_y = self.kf.x[0], self.kf.x[1]
            coords = (predicted_x, predicted_y)
            uncertainty = np.trace(self.kf.P[:2, :2])
            conf = float(np.exp(-0.01 * uncertainty))
            
            return coords, conf

        info_logger.info("[BallTracker] No se pudo recuperar la posicion del balon")
        return None, None

    def _persist_state(self, frame_number: int, conf: float):
        state = self.get_ball_state()
        tools = context.analysis_context.tools

        if state is None:
            return
        tracks_collection = TrackCollectionBall()
        info_logger.info("Actual state to persist: ", state)
        payload = {
            "frame_index": frame_number,
            "x": state["x"],
            "y": state["y"],
            "z": 0.0,
            "vx": state["vx"],
            "vy": state["vy"],
            "bbox": json.dumps(state["bbox"]),
            "owner_id": state["owner_id"],
            "track_id": 0,
        }
        try:
            print("Creating new ball record: ", payload)
            ball_state = tracks_collection.post(payload)
            if ball_state is not None:
                value_store.globals.add_detected_object(DetectedObjectData(
                    name="ball",
                    id=ball_state.id,
                    confidence=conf
                ))
                tools.analysis_data_collector.add_row(AnalysisData(
                    frame=frame_number,
                    track_id=0,
                    x=state["x"],
                    y=state["y"],
                    vclass="ball",
                    timestamps=value_store.globals.timestamp
                ))
        except Exception as e:
            logger.exception(f"Error al guardar el balon en frame {frame_number}: {e}")
            raise e
