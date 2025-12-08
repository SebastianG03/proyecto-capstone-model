import json
import logging
from typing import override

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.entities.collections.track_collections import TrackCollectionBall
from app.entities.interfaces.tracker_base import Tracker
from sqlalchemy.orm import Session


class BallTracker(Tracker):

    def __init__(self, model: YOLO):
        super().__init__(model)

    
    @override
    def reset(self):
        """Resetea el estado interno del tracker."""
        pass
    
    @override
    def get_object_tracks(
        self,
        detection_with_tracks,
        cls_names_inv,
        frame_num,
        detection_supervision,
        db: Session
    ):
        print(f"[BallTracker] get_object_tracks llamado frame {frame_num}")
        self.get_tracker_tracks(
            detection_with_tracks,
            cls_names_inv,
            frame_num,
            detection_supervision,
            db
        )

    def get_tracker_tracks(
        self,
        detection_with_tracks: sv.Detections,
        cls_names_inv: dict[str, int],
        frame_num: int,
        detection_supervision: sv.Detections,
        db: Session
    ):
        tracks_collection = TrackCollectionBall(db)
        print(f"[BallTracker] START get_tracker_tracks frame {frame_num}")

        if detection_with_tracks is None:
            print(f"[BallTracker] No detecciones para frame {frame_num}")
            return

        xyxy = getattr(detection_with_tracks, "xyxy", None)
        class_ids = getattr(detection_with_tracks, "class_id", None)
        if xyxy is None or class_ids is None:
            print(f"[BallTracker] No xyxy o class_ids para frame {frame_num}")
            return

        # normalizar arrays
        try:
            class_ids_arr = np.asarray(class_ids)
            xyxy_arr = np.asarray(xyxy)
        except Exception:
            class_ids_arr = class_ids
            xyxy_arr = xyxy

        ball_class_idx = cls_names_inv.get("ball")
        if ball_class_idx is None:
            print(f"[BallTracker] No hay clase 'ball' en frame {frame_num}")
            return

        try:
            mask = class_ids_arr == ball_class_idx
        except Exception:
            mask = (class_ids_arr == ball_class_idx)

        if not getattr(mask, "any", lambda: False)():
            print(f"[BallTracker] Ningún balón detectado en frame {frame_num}")
            return

        try:
            ball_bbox = xyxy_arr[mask][0].tolist()
        except Exception:
            print(f"[BallTracker] Error extrayendo bbox en frame {frame_num}")
            return

        cx, cy = self._bbox_to_center(ball_bbox)
        print("Bbox balón ", ball_bbox, f"centro ({cx}, {cy})")
        payload = {
            "frame_index": int(frame_num),
            "x": float(cx),
            "y": float(cy),
            "z": 0.0,
            "bbox": json.dumps(ball_bbox),
            "owner_id": None,
            "track_id": 0
        }

        existing = tracks_collection.get_record_for_frame(track_id=0, frame_index=int(frame_num))

        try:
            if existing:
                tracks_collection.patch(int(f'{existing.id}'), payload)
            else:
                tracks_collection.post(payload)
        except Exception as e:
            print(f"[BallTracker] DB error frame {frame_num}: {e}")

        print(f"[BallTracker] END get_tracker_tracks frame {frame_num}")
