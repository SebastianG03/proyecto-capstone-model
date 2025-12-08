import logging
import json
from typing import override

import numpy as np
import supervision as sv
from ultralytics import YOLO
from sqlalchemy.orm import Session

from app.entities.interfaces.tracker_base import Tracker
from app.entities.collections import TrackCollectionPlayer
from app.layers.domain import tracks 

class PlayerTracker(Tracker):

    def __init__(self, model: YOLO):
        super().__init__(model)


    @override
    def reset(self) -> None:
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
        print(f"[PlayerTracker] get_object_tracks llamado frame {frame_num}")
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
        tracks_collection = TrackCollectionPlayer(db)
        print(f"[PlayerTracker] START get_tracker_tracks frame {frame_num}")

        if detection_with_tracks is None:
            print(f"[PlayerTracker] No detecciones para frame {frame_num}")
            return

        xyxy = getattr(detection_with_tracks, "xyxy", None)
        class_ids = getattr(detection_with_tracks, "class_id", None)
        tracker_ids = getattr(detection_with_tracks, "tracker_id", None)
        if xyxy is None or class_ids is None or tracker_ids is None:
            print(f"[PlayerTracker] xyxy, class_ids o tracker_ids faltantes frame {frame_num}")
            return

        try:
            xyxy_arr = np.asarray(xyxy)
            class_ids_arr = np.asarray(class_ids)
            tracker_ids_arr = np.asarray(tracker_ids)
        except Exception:
            xyxy_arr, class_ids_arr, tracker_ids_arr = xyxy, class_ids, tracker_ids

        player_class_idx = cls_names_inv.get("player")
        if player_class_idx is None:
            print(f"[PlayerTracker] No hay clase 'player' en frame {frame_num}")
            return

        try:
            mask = class_ids_arr == player_class_idx
        except Exception:
            mask = (class_ids_arr == player_class_idx)

        if not getattr(mask, "any", lambda: False)():
            print(f"[PlayerTracker] Ningún jugador detectado frame {frame_num}")
            return

        player_bboxes = xyxy_arr[mask]
        player_ids = tracker_ids_arr[mask]


        for bbox_arr, raw_tid in zip(player_bboxes, player_ids):
            if raw_tid is None:
                continue
            try:
                track_id = int(raw_tid)
            except Exception:
                print(f"[PlayerTracker] Track id inválido: {raw_tid}")
                continue
            try:
                bbox_list = bbox_arr.tolist()
            except Exception:
                bbox_list = list(map(float, bbox_arr))

            cx, cy = self._bbox_to_center(bbox_list)
            print("Bbox jugador ", track_id, bbox_list, f"centro ({cx}, {cy})")
            payload = {
                "player_id": track_id,
                "frame_index": int(frame_num),
                "bbox": json.dumps(bbox_list),
                "x": float(cx),
                "y": float(cy),
                "z": 0.0
            }

            print(f"[PlayerTracker] Buscando jugador {track_id} frame {frame_num}")
            existing = tracks_collection.get_record_for_frame(track_id=track_id, frame_index=int(frame_num))

            try:
                if existing:
                    tracks_collection.patch(int(f'{existing.id}'), payload)
                else:
                    tracks_collection.post(payload)
            except Exception as e:
                print(f"[PlayerTracker] DB error jugador {track_id} frame {frame_num}: {e}")

        print(f"[PlayerTracker] END get_tracker_tracks frame {frame_num}")
