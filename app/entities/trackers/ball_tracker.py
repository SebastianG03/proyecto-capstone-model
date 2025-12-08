from asyncio.log import logger
import json
from typing import Any, Dict, Optional, override
from filterpy.kalman import KalmanFilter

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.entities.collections.track_collections import TrackCollectionBall
from app.entities.interfaces.tracker_base import Tracker
from sqlalchemy.orm import Session


class BallTracker(Tracker):

    def __init__(
        self,
        model: YOLO,
        max_age: int = 5,
        gate: float = 9.21,
        df: float = 1.0):
        super().__init__(model)
        self.kf = self._create_kf(df)
        self.age = 0
        self.max_age = max_age
        self.gate = gate  # chi2 2-dof 95%
        self.owner_id: Optional[int] = None
        self.possession_time: float = 0.0
        self.last_seen_frame: int = -1

    def _create_kf(self, dt: float = 1.0) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        kf.R *= 9  # observación
        kf.Q[2:, 2:] *= 0.1  # velocidad
        kf.x = np.array([0., 0., 0., 0.])  # se inicializa en primera detección
        kf.P *= 100
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
        detection_with_tracks,
        cls_names_inv,
        frame_num,
        detection_supervision,
        db: Session
    ):
        ball = self._extract_ball(detection_with_tracks, cls_names_inv)
        self._update_kf(ball, frame_num)
        self._persist_state(frame_num, db)
        logger.debug(f"[BallTracker] frame {frame_num} owner={self.owner_id}")
        
    def _extract_ball(
        self,
        detections: sv.Detections,
        cls_names_inv: dict):
        try:
            print("[BallTracker] extract_ball llamado")
            if detections is None:
                return None
            print("[BallTracker] extract_ball procesando detecciones")
            xyxy = np.asarray(getattr(detections, "xyxy", None))
            class_ids = np.asarray(getattr(detections, "class_id", None))
            if xyxy is None or class_ids is None:
                print("[BallTracker] No xyxy o class_ids")
                return None
            print("[BallTracker] Buscando clase 'ball'")
            ball_idx = cls_names_inv.get("ball")
            if ball_idx is None:
                return None
            mask = class_ids == ball_idx
            if not mask.any():
                print("[BallTracker] Ningún balón detectado")
                return None
            print("[BallTracker] Extrayendo bbox del balón")
            bbox = xyxy[mask][0].tolist()
            print("[BallTracker] Extrayendo centro del balón")
            cx, cy = self._bbox_to_center(bbox)
            print(f"[BallTracker] Centro del balón: ({cx}, {cy})")
            return cx, cy
        except Exception as e:
            print(f"[BallTracker] Error extrayendo balón: {e}")
            return None

    def _update_kf(
        self,
        ball_coordinates: Optional[tuple[float, float]],
        frame_number: int):
        if ball_coordinates is not None:
            z = np.array(ball_coordinates)
            # Gate de Mahalanobis
            y, S = self.kf.y, self.kf.S
            d2 = float(y.T @ np.linalg.inv(S) @ y)
            if d2 < self.gate:
                self.kf.update(z)
                self.age = 0
                self.last_seen_frame = frame_number
            else:
                logger.debug(f"Detección rechazada por gate (d2={d2:.2f})")
        else:
            self.age += 1
        self.kf.predict()

    def get_ball_state(self) -> Optional[Dict[str, Any]]:
        if self.age > self.max_age:
            return None
        x, y = self.kf.x[:2]
        return {
            "x": float(x),
            "y": float(y),
            "vx": float(self.kf.x[2]),
            "vy": float(self.kf.x[3]),
            "owner_id": self.owner_id,
            "possession_time": self.possession_time,
            "last_seen_frame": self.last_seen_frame
        }

    def _persist_state(self, frame_number: int, db: Session):
        state = self.get_ball_state()
        if state is None:
            return
        tracks_collection = TrackCollectionBall(db)
        payload = {
            "frame_index": frame_number,
            "x": state["x"],
            "y": state["y"],
            "z": 0.0,
            "vx": state["vx"],
            "vy": state["vy"],
            "bbox": json.dumps([]),  # no usamos bbox filtrada por ahora
            "owner_id": state["owner_id"],
            "track_id": 0
        }
        existing = tracks_collection.get_record_for_frame(track_id=0, frame_index=frame_number)
        try:
            if existing:
                tracks_collection.patch(int(f'{existing.id}'), payload)
            else:
                tracks_collection.post(payload)
        except Exception as e:
            logger.exception(f"DB error frame {frame_number}: {e}")

    # @override
    # def get_object_tracks(
    #     self,
    #     detection_with_tracks,
    #     cls_names_inv,
    #     frame_num,
    #     detection_supervision,
    #     db: Session
    # ):
        
    #     print(f"[BallTracker] get_object_tracks llamado frame {frame_num}")
    #     tracks_collection = TrackCollectionBall(db)
    #     print(f"[BallTracker] START get_tracker_tracks frame {frame_num}")

    #     if detection_with_tracks is None:
    #         print(f"[BallTracker] No detecciones para frame {frame_num}")
    #         return

    #     xyxy = getattr(detection_with_tracks, "xyxy", None)
    #     class_ids = getattr(detection_with_tracks, "class_id", None)
    #     if xyxy is None or class_ids is None:
    #         print(f"[BallTracker] No xyxy o class_ids para frame {frame_num}")
    #         return

    #     # normalizar arrays
    #     try:
    #         class_ids_arr = np.asarray(class_ids)
    #         xyxy_arr = np.asarray(xyxy)
    #     except Exception:
    #         class_ids_arr = class_ids
    #         xyxy_arr = xyxy

    #     ball_class_idx = cls_names_inv.get("ball")
    #     if ball_class_idx is None:
    #         print(f"[BallTracker] No hay clase 'ball' en frame {frame_num}")
    #         return

    #     try:
    #         mask = class_ids_arr == ball_class_idx
    #     except Exception:
    #         mask = (class_ids_arr == ball_class_idx)

    #     if not getattr(mask, "any", lambda: False)():
    #         print(f"[BallTracker] Ningún balón detectado en frame {frame_num}")
    #         return

    #     try:
    #         ball_bbox = xyxy_arr[mask][0].tolist()
    #     except Exception:
    #         print(f"[BallTracker] Error extrayendo bbox en frame {frame_num}")
    #         return

    #     cx, cy = self._bbox_to_center(ball_bbox)
    #     print("Bbox balón ", ball_bbox, f"centro ({cx}, {cy})")
    #     payload = {
    #         "frame_index": int(frame_num),
    #         "x": float(cx),
    #         "y": float(cy),
    #         "z": 0.0,
    #         "bbox": json.dumps(ball_bbox),
    #         "owner_id": None,
    #         "track_id": 0
    #     }

    #     existing = tracks_collection.get_record_for_frame(track_id=0, frame_index=int(frame_num))

    #     try:
    #         if existing:
    #             tracks_collection.patch(int(f'{existing.id}'), payload)
    #         else:
    #             tracks_collection.post(payload)
    #     except Exception as e:
    #         print(f"[BallTracker] DB error frame {frame_num}: {e}")

    #     print(f"[BallTracker] END get_tracker_tracks frame {frame_num}")