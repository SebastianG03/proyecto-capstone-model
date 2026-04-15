import json
import math
from typing import Any, List, Tuple, override

import numpy as np
from rich.progress import track
import supervision as sv
from ultralytics.models import YOLO
from sqlalchemy.orm import Session

from app.entities.interfaces.tracker_base import Tracker
from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState
import app.entities.utils.global_values_store as value_store
from app.entities.models.detected_object_data import AnalysisData, DetectedObjectData
import app.entities.utils.tools_context as context
from app.logger import info_logger

class State:
    def __init__(
        self,
        bbox: list[float],
        x: float,
        y: float,
        conf: float,
        timestamp: float,
        player_id: int,
        frame_num: int):
        self.bbox = bbox
        self.x = x
        self.y = y
        self.conf = conf
        self.timestamp = timestamp
        self.player_id = player_id
        self.frame_num = frame_num
        

class PlayerTracker(Tracker):
    def __init__(self, model: YOLO):
        super().__init__(model)
        self.movement_threshold = 15

    @override
    def reset(self) -> None:
        pass

    @override
    def get_object_tracks(
        self,
        detections: List[sv.Detections],
        cls_names_inv,
        frame_num,
        db: Session,
    ):
        print(f"[PlayerTracker] get_object_tracks llamado frame {frame_num}")
        self.get_tracker_tracks(detections, cls_names_inv, frame_num)
    
    def calculate_iou(self, bboxA: list[float], bboxB: list[float]) -> float:
        xA = max(bboxA[0], bboxB[0])
        yA = max(bboxA[1], bboxB[1])
        xB = min(bboxA[2], bboxB[2])
        yB = min(bboxA[3], bboxB[3])
        
        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        
        areaA = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
        areaB = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
        
        union =  areaA + areaB - inter_area
        return inter_area / union if union > 0 else 0

    def extract_tracker_data(
        self,
        detections: List,
        ) -> List[Tuple[list[float], float]]:

        valid_detections = []

        for det in detections:
            if det.boxes is not None:
                for box in det.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    info_logger.info(f"[PlayerTracker] Informacion dentro del box: {box}")

                    if bbox is None or len(bbox) != 4:
                            info_logger.info("[PlayerTracker] BBox inválida, omitiendo.")
                            continue

                    valid_detections.append((bbox, conf))

        return valid_detections
        

    def get_tracker_tracks(
        self,
        detection_with_tracks: List,
        cls_names_inv: dict[str, int],
        frame_num: int,
    ):
        tools = context.analysis_context.tools
        tracks_collection = TrackCollectionPlayer()
        self.logger.info(f"[PlayerTracker] START get_tracker_tracks frame {frame_num}")

        if detection_with_tracks is None:
            self.logger.info(f"[PlayerTracker] No detecciones para frame {frame_num}")
            return

        payloads = self.validate_unique_players(detection_with_tracks, cls_names_inv, frame_num, value_store.globals.timestamp)
        
        if payloads is None or len(payloads) == 0:
            self.logger.info(f"[PlayerTracker] No hay detecciones de jugadores únicas para frame {frame_num}")
            return

        payloads = self.validate_previous_players(frame_num, list(payloads.values()))
        
        for payload in payloads:
            player_id = payload["player_id"]
            player_exists = tracks_collection.verify_player_exists(player_id)
            state_exist = tracks_collection.verify_player_state_exists(
                player_id, frame_num
            )
            
            player_payload = {
                "player_id": player_id,
            }
            
            info_logger.info(f"[PlayerTracker] Procesando player_id {player_id} en frame {frame_num} con payload {payload}, player payload: {player_payload}, player_exists: {player_exists}, state_exist: {state_exist}")
            
            if not player_exists:
                tracks_collection.post(player_payload)
            
            if state_exist:
                state = tracks_collection.patch_state(player_id, frame_num, payload)
                self.update_global_state_id(state, payload["conf"])
            else:
                state = tracks_collection.post_state(payload)
                self.update_global_state_id(state, payload["conf"])
                tools.analysis_data_collector.add_row(AnalysisData(
                    frame=frame_num,
                    track_id=player_id,
                    x=payload["x"],
                    y=payload["y"],
                    vclass="player",
                    timestamps=value_store.globals.timestamp,
                ))

    def update_global_state_id(self, state: PlayerState | None, conf: float):
        if state is not None:
            id = int(f"{state.id}")
            name = "player"
            value_store.globals.add_detected_object(DetectedObjectData(name=name, id=id, confidence=conf))

    def validate_unique_players(self, detections: List, cls_names_inv: dict[str, int], frame_num: int, timestamp: float):
        unique_detections: dict[int, dict[str, Any]] = {} # track_id -> Payload
        det_data = self.extract_tracker_data(detections)

        if det_data is None or len(det_data) == 0:
            return
        
        bboxs: list[list[float]] = []
        confs: list[float] = []
        
        for bbox, conf in det_data:
            
            bboxs.append(bbox)
            confs.append(conf)
        
        xyxy = np.array(bboxs)
        confs_arr = np.array(confs)
        
        sv_detections = sv.Detections(xyxy=xyxy, confidence=confs_arr, class_id=np.zeros(len(det_data)))
        tracked = self.tracker.update_with_detections(sv_detections)
        
        if tracked is None or len(tracked) == 0:
            return
        
        track_ids = np.asarray(tracked.tracker_id)
        bbox = np.asarray(tracked.xyxy)
        conf = np.asarray(tracked.confidence)

        for track_id, bbox, conf in zip(track_ids, bbox, conf):
            
            track_id = int(track_id)
            bbox_list = bbox.tolist()
            cx, cy = self._bbox_to_center(bbox_list)
            
            if track_id not in unique_detections:
                unique_detections[track_id] = {
                    "player_id": track_id,
                    "frame_index": int(frame_num),
                    "bbox": bbox_list,
                    "x": float(cx),
                    "y": float(cy),
                    "z": 0.0,
                    "timestamp_ms": timestamp,
                    "conf": float(conf),
                }
            else:
                prev = unique_detections[track_id]
                if conf > prev["conf"]:
                    unique_detections[track_id] = {
                        "player_id": track_id,
                        "frame_index": int(frame_num),
                        "bbox": bbox_list,
                        "x": float(cx),
                        "y": float(cy),
                        "z": 0.0,
                        "timestamp_ms": timestamp,
                        "conf": float(conf),
                    }

        items = list(unique_detections.items())
        to_remove = set()
        
        for track_id, detection in items:
            bbox_list = detection["bbox"]
            cx, cy = detection["x"], detection["y"]
            conf = detection["conf"]

            for existing_id, existing_detection in items:
                if track_id == existing_id:
                    continue

                next_bbox_list = existing_detection["bbox"]
                next_cx, next_cy = existing_detection["x"], existing_detection["y"]

                distance = np.linalg.norm(np.array([cx, cy]) - np.array([next_cx, next_cy]))

                if distance > self.movement_threshold:
                    continue
                
                iou = self.calculate_iou(bbox_list, next_bbox_list)
                info_logger.info(f"[PlayerTracker] Validando detecciones {track_id} y {existing_id} con distancia {distance:.4f} y IoU {iou:.4f} en frame {frame_num}")
                
                if 0.3 <= iou < 0.5:
                    continue

                if iou > 0.5:
                    info_logger.info(f"[PlayerTracker] Detecciones {track_id} y {existing_id} consideradas iguales por IoU en frame {frame_num}")
                    existing_conf = unique_detections[existing_id]["conf"]
                    if conf > existing_conf:
                        to_remove.add(existing_id)
                    else:
                        to_remove.add(track_id)

        for rid in to_remove:
            unique_detections.pop(rid, None)
        
        for player_id, detection in list(unique_detections.items()):
            unique_detections[player_id]["bbox"] = json.dumps(detection["bbox"])

        return unique_detections

    def validate_previous_players(self, frame_num, state_payloads: List[dict]):
        if frame_num <= value_store.globals.fps * 2:
            info_logger.info(f"[PlayerTracker] Insuficientes frames para validar estados anteriores en frame {frame_num}")
            return state_payloads
        
        previous_states = context.analysis_context.tools.player_records.get_states_previous_frame(frame_num)
        
        if previous_states is None or len(previous_states) == 0:
            info_logger.info(f"[PlayerTracker] No hay estados anteriores para validar en frame {frame_num}")
            return state_payloads
        
        previous_states = [state.to_dict() for state in previous_states]
        previous_states.sort(key=lambda s: s["frame_index"], reverse=True)
        unique_prev = set()
        
        for state in previous_states:
            pid = state["player_id"]
            
            if state["frame_index"] - frame_num > value_store.globals.fps * 1.5:
                    continue

            if pid not in unique_prev:
                unique_prev.add(state)
        
        previous_states = [state for state in previous_states if state in unique_prev]

        used_prev_ids = set()
        assigned_payloads = set()
        matches = []
        
        for i, state_payload in enumerate(state_payloads):
            x = float(state_payload["x"])
            y = float(state_payload["y"])

            for state in previous_states:
                prev_x, prev_y = float(state["x"]), float(state["y"])
                prev_player_id = int(state["player_id"])

                distance = np.linalg.norm(np.array([prev_x, prev_y]) - np.array([x, y]))

                if distance < self.movement_threshold:
                    matches.append((distance, i, prev_player_id))
        
        matches.sort(key=lambda x: x[0])
        
        for dist, payload_idx, prev_id in matches:
            if payload_idx not in assigned_payloads:
                info_logger.info(f"[PlayerTracker] Nuevo jugador en frame {frame_num} con player_id {state_payloads[payload_idx]['player_id']} cerca de player_id {prev_id} del frame anterior con distancia {dist:.4f}")

            if payload_idx in assigned_payloads or prev_id in used_prev_ids:
                continue
            
            payload = state_payloads[payload_idx]
            current_id = payload["player_id"]
            
            if current_id != prev_id:
                info_logger.info(f"[PlayerTracker] Reasignando player_id {current_id} a {prev_id} por distancia {dist:.4f} en frame {frame_num}")
                payload["player_id"] = prev_id

            assigned_payloads.add(payload_idx)
            used_prev_ids.add(prev_id)
        
        return state_payloads
    
    
    def validate_players(self, state_a: State, state_b: State):
        xo = np.array([state_a.x, state_a.y])
        xf = np.array([state_b.x, state_b.y])
        time_sec = abs(state_b.timestamp - state_a.timestamp) * 0.001 
        
        delta_x = xo - xf
        vo = delta_x / time_sec
        acceleration = (2 * (delta_x - vo * time_sec)) / time_sec ** 2
        vf = vo + acceleration * time_sec
        direction = math.atan2(delta_x[1], delta_x[0])
        
        if delta_x[1] < 0 and delta_x[0] < 0:
            direction += 180
        elif delta_x[1] > 0 and delta_x[0] < 0:
            direction = 180 - direction
        elif delta_x[1] < 0 and delta_x[0] > 0:
            direction = 360 - direction

        
        
        