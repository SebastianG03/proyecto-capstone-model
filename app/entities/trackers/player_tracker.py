import json
import math
from typing import Any, List, Tuple, override

import joblib
import numpy as np
from ultralytics.models import YOLO
from sqlalchemy.orm import Session

from app.entities.interfaces.tracker_base import Tracker
from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState, State, PlayerStatus
from app.entities.trackers.player.player_matcher import PlayerMatcher
import app.entities.utils.global_values_store as value_store
from app.entities.models.detected_object_data import AnalysisData, DetectedObjectData
from app.entities.trackers.player.player_kalman_filter import PlayerKalmanFilter
import app.entities.utils.tools_context as context
from app.logger import info_logger
import xgboost as xgb

from app.utils.routes import PLAYER_XGB_MODEL

class PlayerTracker(Tracker):
    def __init__(self, model: YOLO):
        super().__init__(model)
        self.player_matcher = PlayerMatcher()

    @override
    def get_object_tracks(
        self,
        detections: List,
        cls_names_inv,
        frame_num,
        db: Session,
    ):
        self.logger.info(f"[PlayerTracker] START get_tracker_tracks frame {frame_num}")

        if detections is None:
            self.logger.info(f"[PlayerTracker] No detecciones para frame {frame_num}")
            return

        payloads = self.validate_unique_players(detections, frame_num, value_store.globals.timestamp)
        
        if payloads is None or len(payloads) == 0:
            self.logger.info(f"[PlayerTracker] No hay detecciones de jugadores unicas para frame {frame_num}")
            detected_ids = set(payload["player_id"] for payload in payloads.values())
            predicted_payloads = self.player_matcher.handle_missed_players(frame_num, detected_ids)
            self.save_payloads(predicted_payloads, frame_num)
            return
        
        
        payloads = self.player_matcher.validate_previous_players(frame_num, list(payloads.values()))
        
        payloads_states = [State.to_instance(payload) for payload in payloads]

        for state in payloads_states:
            self.player_matcher.update_kalman(state) 
        
        payloads_by_player_id = self.group_states_by_player_id(payloads_states)
        
        for player_id, states in payloads_by_player_id.items():
            player_statuses = []
            states.sort(key=lambda x: x.frame_num)
            max_states = len(states)
            if max_states == 1:
                continue

            for i in range(max_states):
                if i + 1 >= max_states:
                    break

                status = self.player_matcher.calculate_player_status(states[i], states[i + 1])
                player_statuses.append(status)

            for i in range(len(player_statuses)):
                if i + 1 >= len(player_statuses):
                    break

                acceleration = self.player_matcher.calculate_state_acceleration(player_statuses[i], player_statuses[i + 1])
            
                if player_id not in self.player_matcher.kalman_filters:
                    kalman_pred = None, None
                else:
                    kf = self.player_matcher.kalman_filters[player_id]
                    kalman_pred = kf.predict()

                if kalman_pred[0] is None or kalman_pred[1] is None:
                    continue
                
                features = self.player_matcher.build_features(player_statuses[i], states, acceleration, kalman_pred)

                if features is None:
                    continue

                # score = self.player_matcher.predict_reid_probability(features)

                # if score > 0.6:
                #     info_logger.info(f"[PlayerTracker] Reidentificando player_id con score {score}")

        self.save_payloads(payloads, frame_num)


    def extract_tracker_data(
        self,
        detections: List,
        ) -> List[Tuple[list[float], float, int]]:

        valid_detections = []

        for det in detections:
            if det.boxes is not None:
                
                boxes = det.boxes
                ids = boxes.id
                xyxy = boxes.xyxy
                confs = boxes.conf
                
                if ids is None:
                    info_logger.info("[PlayerTracker] No se encontraron IDs en las detecciones, omitiendo.")
                    continue
                
                for i in range(len(ids)):
                    track_id = int(ids[i].item())
                    bbox = xyxy[i].cpu().numpy().tolist()
                    conf = float(confs[i].item())
                    
                    valid_detections.append((bbox, conf, track_id))

        return valid_detections

    def save_payloads(self, payloads: list[dict], frame_num: int):
        tracks_collection = TrackCollectionPlayer()
        tools = context.analysis_context.tools
        trained = self.player_matcher.train_from_buffer()

        if trained:
            info_logger.info("[PlayerTracker] Modelo ML actualizado al final del partido.")
        else:
            info_logger.info("[PlayerTracker] No se entrenó — buffer insuficiente, se acumula para el próximo partido.")

        for payload in payloads:
            player_id = payload["player_id"]
            player_exists = tracks_collection.verify_player_exists(player_id)
            state_exist = tracks_collection.verify_player_state_exists(
                player_id, frame_num
            )
            payload["bbox"] = json.dumps(payload["bbox"])
            
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
                
                
    def group_states_by_player_id(self, states: List[State]) -> dict[int, List[State]]:
        states_by_player_id = {}
        for state in states:
            player_id = state.player_id
            if player_id not in states_by_player_id:
                states_by_player_id[player_id] = []
            states_by_player_id[player_id].append(state)
        return states_by_player_id

    def update_global_state_id(self, state: PlayerState | None, conf: float):
        if state is not None:
            id = int(f"{state.id}")
            name = "player"
            value_store.globals.add_detected_object(DetectedObjectData(name=name, id=id, confidence=conf))

    def validate_unique_players(self, detections: List, frame_num: int, timestamp: float):
        unique_detections: dict[int, dict[str, Any]] = {} # track_id -> Payload
        det_data = self.extract_tracker_data(detections)

        if det_data is None or len(det_data) == 0:
            return {}

        for bbox, conf, track_id in det_data:
            track_id = int(track_id)
            cx, cy = self._bbox_to_center(bbox)

            unique_detections[track_id] = {
                "player_id": track_id,
                "frame_index": int(frame_num),
                "bbox": bbox,
                "x": float(cx),
                "y": float(cy),
                "z": 0.0,
                "timestamp_ms": timestamp,
                "conf": float(conf),
            }

        items = list(unique_detections.items())
        to_remove = set()

        for i in range(len(items)):
            track_id, detection = items[i]

            bbox_list = detection["bbox"]
            cx, cy = detection["x"], detection["y"]
            conf = detection["conf"]

            for j in range(i + 1, len(items)):
                existing_id, existing_detection = items[j]

                if existing_id == track_id:
                    continue

                next_bbox_list = existing_detection["bbox"]
                next_cx, next_cy = existing_detection["x"], existing_detection["y"]

                distance = np.linalg.norm(np.array([cx, cy]) - np.array([next_cx, next_cy]))

                iou = self.calculate_iou(bbox_list, next_bbox_list)
                info_logger.info(f"[PlayerTracker] Validando detecciones {track_id} y {existing_id} con distancia {distance:.4f} y IoU {iou:.4f} en frame {frame_num}")
                
                if track_id not in self.player_matcher.kalman_filters:
                    mahalanobis_score = np.inf
                else:
                    kf = self.player_matcher.kalman_filters[track_id]
                    mahalanobis_score = kf.mahalanobis_distance(cx, cy)

                info_logger.info(f"[PlayerTracker] Mahalanobis score en validate unique players: {mahalanobis_score}")

                if (
                    iou > 0.6 and
                    distance < self.player_matcher.movement_threshold and
                    mahalanobis_score < self.player_matcher.CHI_GATE):
                    info_logger.info(f"[PlayerTracker] Detecciones {track_id} y {existing_id} consideradas iguales por IoU en frame {frame_num}")
                    existing_conf = unique_detections[existing_id]["conf"]
                    if conf > existing_conf:
                        to_remove.add(existing_id)
                    else:
                        to_remove.add(track_id)

        for rid in to_remove:
            unique_detections.pop(rid, None)

        return unique_detections
