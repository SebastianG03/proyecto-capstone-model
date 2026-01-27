import json
from typing import override

import numpy as np
import supervision as sv
from ultralytics.models import YOLO
from sqlalchemy.orm import Session

from app.entities.interfaces.tracker_base import Tracker
from app.entities.collections import TrackCollectionPlayer
from app.entities.utils.global_values_store import GlobalValuesStore
from app.logger import info_logger


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
        db: Session,
    ):
        print(f"[PlayerTracker] get_object_tracks llamado frame {frame_num}")
        self.get_tracker_tracks(detection_with_tracks, cls_names_inv, frame_num, db)

    def get_tracker_tracks(
        self,
        detection_with_tracks: sv.Detections,
        cls_names_inv: dict[str, int],
        frame_num: int,
        db: Session,
    ):
        tracks_collection = TrackCollectionPlayer(db)
        info_logger.info(f"[PlayerTracker] START get_tracker_tracks frame {frame_num}")

        if detection_with_tracks is None:
            info_logger.info(f"[PlayerTracker] No detecciones para frame {frame_num}")
            return

        xyxy = getattr(detection_with_tracks, "xyxy", None)
        class_ids = getattr(detection_with_tracks, "class_id", None)
        tracker_ids = getattr(detection_with_tracks, "tracker_id", None)
        info_logger.info(f"[PlayerTracker] xyxy: {xyxy}, class_ids: {class_ids}, tracker_ids: {tracker_ids}")
        if xyxy is None or class_ids is None or tracker_ids is None:
            info_logger.info(
                f"[PlayerTracker] xyxy, class_ids o tracker_ids faltantes frame {frame_num}"
            )
            return

        try:
            xyxy_arr = np.asarray(xyxy)
            class_ids_arr = np.asarray(class_ids)
            tracker_ids_arr = np.asarray(tracker_ids)
        except Exception:
            xyxy_arr, class_ids_arr, tracker_ids_arr = xyxy, class_ids, tracker_ids

        player_class_idx = cls_names_inv.get("player")
        info_logger.info(f"[PlayerTracker] player_class_idx: {player_class_idx}")
        if player_class_idx is None:
            info_logger.info(
                f"[PlayerTracker] No hay clase 'player' en frame {frame_num}"
            )
            return

        mask = class_ids_arr == player_class_idx
        info_logger.info(f"[PlayerTracker] mask: {mask}")

        player_bboxes = xyxy_arr[mask]
        player_ids = tracker_ids_arr[mask]

        for bbox_arr, raw_tid in zip(player_bboxes, player_ids):
            if raw_tid is None or bbox_arr is None:
                continue
            try:
                track_id = int(raw_tid)
            except Exception:
                info_logger.info(f"[PlayerTracker] Track id inválido: {raw_tid}")
                continue
            try:
                bbox_list = bbox_arr.tolist()
            except Exception:
                bbox_list = list(map(float, bbox_arr))

            cx, cy = self._bbox_to_center(bbox_list)
            info_logger.info(f"Bbox jugador {track_id} {bbox_list} centro ({cx}, {cy})")
            timestamp = GlobalValuesStore().timestamp
            info_logger.info(f"[PlayerTracker] Timestamp: {timestamp}")
            payload = {
                "player_id": track_id,
                "frame_index": int(frame_num),
                "bbox": json.dumps(bbox_list),
                "x": float(cx),
                "y": float(cy),
                "z": 0.0,
                "timestamp_ms": timestamp,
            }
            info_logger.info(f"[PlayerTracker] Payload generado {payload}")

            info_logger.info(
                f"[PlayerTracker] Buscando jugador {track_id} frame {frame_num}"
            )
            existing = tracks_collection.verify_player_exists(track_id)
            state_exist = tracks_collection.verify_player_state_exists(
                track_id, frame_num
            )

            if not existing:
                tracks_collection.post({"player_id": track_id})
                tracks_collection.post_state(payload)
            if existing and state_exist:
                tracks_collection.patch_state(track_id, frame_num, payload)
            elif existing and not state_exist:
                tracks_collection.post_state(payload)
        print(f"[PlayerTracker] END get_tracker_tracks frame {frame_num}")
