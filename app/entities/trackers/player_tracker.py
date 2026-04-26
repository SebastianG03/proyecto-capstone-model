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
import app.entities.utils.global_values_store as value_store
from app.entities.models.detected_object_data import AnalysisData, DetectedObjectData
from app.entities.utils.player_kalman_filter import PlayerKalmanFilter
import app.entities.utils.tools_context as context
from app.logger import info_logger
import xgboost as xgb

from app.utils.routes import PLAYER_XGB_MODEL

class PlayerTracker(Tracker):
    def __init__(self, model: YOLO):
        super().__init__(model)
        self.movement_threshold = 15
        self.kalman_filters: dict[int, PlayerKalmanFilter] = {}
        self.last_timestamps: dict[int, float] = {}
        self.CHI_GATE = np.sqrt(5.991)
        self.id_switch_votes = {}
        self.player_missed_counts: dict[int, int] = {} # track_id -> missed_count
        self.MAX_SPEED = 30

        player_model_exists = (PLAYER_XGB_MODEL / "player_model.pkl").exists()
        
        if player_model_exists:
            self.ml_model = joblib.load((PLAYER_XGB_MODEL / "player_model.pkl").as_posix())
            self.ml_model.set_params(n_estimators=200, max_depth=6, learning_rate=0.05)
        else:
            self.ml_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                objective="binary:logistic"
            )


    @override
    def get_object_tracks(
        self,
        detections: List,
        cls_names_inv,
        frame_num,
        db: Session,
    ):
        print(f"[PlayerTracker] get_object_tracks llamado frame {frame_num}")
        self.get_tracker_tracks(detections, cls_names_inv, frame_num)
    

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

    def get_tracker_tracks(
        self,
        detection_with_tracks: List,
        cls_names_inv: dict[str, int],
        frame_num: int,
    ):
        self.logger.info(f"[PlayerTracker] START get_tracker_tracks frame {frame_num}")

        if detection_with_tracks is None:
            self.logger.info(f"[PlayerTracker] No detecciones para frame {frame_num}")
            return

        payloads = self.validate_unique_players(detection_with_tracks, cls_names_inv, frame_num, value_store.globals.timestamp)
        
        if payloads is None or len(payloads) == 0:
            self.logger.info(f"[PlayerTracker] No hay detecciones de jugadores unicas para frame {frame_num}")
            predicted_payloads = self.handle_missed_players(frame_num)
            self.save_payloads(predicted_payloads, frame_num)
            return

        payloads = self.validate_previous_players(frame_num, list(payloads.values()))
        
        payloads_states = [State.to_instance(payload) for payload in payloads]

        for state in payloads_states:
            self.update_kalman(state) 
        
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

                status = self.calculate_player_status(states[i], states[i + 1])
                player_statuses.append(status)

            for i in range(len(player_statuses)):
                if i + 1 >= len(player_statuses):
                    break

                acceleration = self.calculate_state_acceleration(player_statuses[i], player_statuses[i + 1])
                kalman_pred = self.predict_player_position(player_statuses[i].player_id)

                if kalman_pred[0] is None or kalman_pred[1] is None:
                    continue
                
                features = self.build_features(player_statuses[i], states, acceleration, kalman_pred)

                if features is None:
                    continue

                score = self.predict_reid_probability(features)

                if score > 0.6:
                    info_logger.info(f"[PlayerTracker] Reidentificando player_id con score {score}")

        self.save_payloads(payloads, frame_num)


    def save_payloads(self, payloads: list[dict], frame_num: int):
        tracks_collection = TrackCollectionPlayer()
        tools = context.analysis_context.tools

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

    def validate_unique_players(self, detections: List, cls_names_inv: dict[str, int], frame_num: int, timestamp: float):
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
                mahalanobis_score = self.mahalanobis_distance(track_id, cx, cy)
                info_logger.info(f"[PlayerTracker] Mahalanobis score en validate unique players: {mahalanobis_score}")

                if (
                    iou > 0.6 and
                    distance < self.movement_threshold and
                    mahalanobis_score < self.CHI_GATE):
                    info_logger.info(f"[PlayerTracker] Detecciones {track_id} y {existing_id} consideradas iguales por IoU en frame {frame_num}")
                    existing_conf = unique_detections[existing_id]["conf"]
                    if conf > existing_conf:
                        to_remove.add(existing_id)
                    else:
                        to_remove.add(track_id)

        for rid in to_remove:
            unique_detections.pop(rid, None)

        return unique_detections
    
    def filter_previous_states(self, frame_num: int) -> List[dict]:
        previous_states = context.analysis_context.tools.player_records.get_states_previous_frame(frame_num)

        if previous_states is None or len(previous_states) == 0:
            info_logger.info(f"[PlayerTracker] No hay estados anteriores para validar en frame {frame_num}")
            return []

        previous_states = [state.to_dict() for state in previous_states]
        previous_states.sort(key=lambda s: s["frame_index"], reverse=True)
        
        return previous_states

    def get_unique_previous_states(self, states: List[dict], frame_num: int) -> List[dict]:
        unique_prev = set()
        filtered_states = []

        for state in states:
            pid = state["player_id"]

            if frame_num - int(state["frame_index"]) > value_store.globals.fps * 1.5:
                    continue
            
            if self.player_missed_counts.get(pid, 0) > 5:
                continue

            if pid not in unique_prev:
                unique_prev.add(pid)
                filtered_states.append(state)
        
        return filtered_states
    
    def generate_candidates(self, payload: dict, previous_states: List[dict]):
        candidates = []
        
        x = float(payload["x"])
        y = float(payload["y"])
        
        for state in previous_states:
            player_id = state["player_id"]
            state_x, state_y = float(state["x"]), float(state["y"])
            maha_dist = self.mahalanobis_distance(player_id, x, y)

            dist = np.linalg.norm(
                np.array([x, y]) -
                np.array([state_x, state_y])
            )
            
            dt = (
                float(payload["timestamp_ms"]) -
                float(state["timestamp_ms"])
            ) * 0.001
            
            if maha_dist > self.CHI_GATE or dist > self.movement_threshold * 3 or dt <= 0:
                continue

            implied_speed = dist / dt

            if implied_speed > self.MAX_SPEED:
                continue

            candidates.append({
                (player_id, maha_dist, dist)
            })
            
        return candidates

    def compute_match_cost(
        self,
        player_id: int,
        payload: dict,
        prev_state: dict
    ):
        payload_x, payload_y = float(payload["x"]), float(payload["y"])
        maha_dist = self.mahalanobis_distance(player_id, payload_x, payload_y)
        dist = np.linalg.norm(
            np.array([payload_x, payload_y]) -
            np.array([prev_state["x"], prev_state["y"]])
        )
        
        pred_x, pred_y = self.predict_player_position(player_id)
        residual = np.linalg.norm(
            np.array([pred_x, pred_y]) -
            np.array([payload_x, payload_y])
        )
        
        return (
            .5 * maha_dist +
            .3 * dist +
            .2 * residual
        )        

    def validate_previous_players(self, frame_num, state_payloads: List[dict]):
        if frame_num <= value_store.globals.fps * 2:
            info_logger.info(f"[PlayerTracker] Insuficientes frames para validar estados anteriores en frame {frame_num}")
            return state_payloads

        previous_states = self.filter_previous_states(frame_num)
        
        if len(previous_states) == 0:
            return state_payloads

        filtered_states = self.get_unique_previous_states(previous_states, frame_num)
        matches = []

        for i, state_payload in enumerate(state_payloads):
            x = float(state_payload["x"])
            y = float(state_payload["y"])

            for state in filtered_states:
                prev_x, prev_y = float(state["x"]), float(state["y"])
                prev_player_id = int(state["player_id"])

                distance = np.linalg.norm(np.array([prev_x, prev_y]) - np.array([x, y]))
                
                score = self.mahalanobis_distance(prev_player_id, x, y)

                info_logger.info(f"[PlayerTracker] Comparando player_id {state_payload['player_id']} con player_id {prev_player_id} con distancia {distance:.4f} en frame {frame_num}, score {score:.4f} vs chi gate {self.CHI_GATE}")
                # TODO validar valores de score
                if score > self.CHI_GATE or np.isinf(score):
                    continue

                matches.append((score, i, prev_player_id))

        matches.sort(key=lambda x: x[0])
        used_prev_ids = set()
        assigned_payloads = set()

        for score, payload_idx, prev_id in matches:
            if payload_idx not in assigned_payloads:
                info_logger.info(f"[PlayerTracker] Nuevo jugador en frame {frame_num} con player_id {state_payloads[payload_idx]['player_id']} cerca de player_id {prev_id} del frame anterior con distancia {score:.4f}")

            if payload_idx in assigned_payloads or prev_id in used_prev_ids:
                continue
            
            payload = state_payloads[payload_idx]
            current_id = payload["player_id"]
            
            if current_id != prev_id:
                info_logger.info(f"[PlayerTracker] Reasignando player_id {current_id} a {prev_id} por score {score:.4f} en frame {frame_num}")
                payload["player_id"] = prev_id

            assigned_payloads.add(payload_idx)
            used_prev_ids.add(prev_id)
        
        predicted_payloads = self.handle_missed_players(frame_num=frame_num)
        validated_payloads = self.validate_payloads(detected_players=state_payloads, predicted_players=predicted_payloads)

        return validated_payloads
    
    def mahalanobis_distance(self, player_id: int, observed_x: float, observed_y: float):
        if player_id not in self.kalman_filters:
            return np.inf
        
        kf = self.kalman_filters[player_id]
        
        H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])
        
        z = np.array([
            [observed_x],
            [observed_y]
        ])
        
        x_pred, _ = self.predict_player_position(player_id)

        if x_pred is None:
            return np.inf

        innovation = z - H @ x_pred
        S = H @ kf.P @ H.T + kf.R
        
        S_inv = np.linalg.pinv(S)
        
        d2 = innovation.T @ S_inv @ innovation

        return float(np.sqrt(d2.item()))
    
    def calculate_player_status(self, state_a: State, state_b: State):
        
        if state_a.player_id != state_b.player_id:
            raise ValueError("Los estados deben pertenecer al mismo jugador para calcular el estado del jugador.")

        xo = np.array([state_a.x, state_a.y])
        xf = np.array([state_b.x, state_b.y])
        time_sec = abs(state_b.timestamp - state_a.timestamp) * 0.001 
        
        delta_x = xf - xo
        
        vx = delta_x[0] / time_sec
        vy = delta_x[1] / time_sec
        speed = np.sqrt(vx**2 + vy**2)
        direction = math.atan2(delta_x[1], delta_x[0])
        direction = math.degrees(direction)

        return PlayerStatus(
            player_id=state_b.player_id,
            frame_index=state_b.frame_num,
            vx=vx,
            vy=vy,
            speed=speed,
            direction=direction,
            time=state_b.timestamp,
            delta_x=delta_x,
            xo=xo,
            xf=xf
        )

    def calculate_state_acceleration(self, status_a: PlayerStatus, status_b: PlayerStatus):
        if status_a.player_id != status_b.player_id:
            raise ValueError("Los estados deben pertenecer al mismo jugador para calcular la aceleracion.")
        
        delta_vx = status_b.vx - status_a.vx
        delta_vy = status_b.vy - status_a.vy
        delta_time = abs(status_b.time - status_a.time)
        
        if delta_time == 0:
            return np.asarray([0, 0])
        
        ax = delta_vx / delta_time
        ay = delta_vy / delta_time
        return np.asarray([ax, ay])

    def update_kalman(self, state: State):
        player_id = state.player_id
        current_time = state.timestamp

        if player_id not in self.kalman_filters:
            kf = PlayerKalmanFilter(dt=1/30)  # fallback
            kf.x[:2] = np.array([[state.x], [state.y]])
            self.kalman_filters[player_id] = kf
            self.last_timestamps[player_id] = current_time
            return kf.x

        kf = self.kalman_filters[player_id]

        prev_time = self.last_timestamps[player_id]
        dt = (current_time - prev_time) * 0.001

        if dt <= 0:
            dt = 1/30

        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])

        kf.predict()

        z = np.array([state.x, state.y])
        kf.update(z)

        self.last_timestamps[player_id] = current_time

        return kf.x

    def predict_player_position(self, player_id: int):
        if player_id not in self.kalman_filters:
            info_logger.info(f"[PlayerTracker] No hay filtro de Kalman para player_id {player_id}, no se puede predecir posicion.")  
            return None, None

        kf = self.kalman_filters[player_id]
        
        x_pred = kf.F @ kf.x
        P_pred = (
            kf.F @ kf.P @ kf.F.T
            + kf.Q
        )
        
        return x_pred, P_pred
    
    def build_features(
        self,
        state: PlayerStatus,
        history: list[State],
        acceleration: np.ndarray,
        kalman_pred: Tuple[np.ndarray, np.ndarray]):
        WINDOWS_SIZE =  30
        features = []
        
        if len(history) < 2:
            return None

        last = history[-1]
        x, y = last.x, last.y
        vx = state.vx
        vy = state.vy
        pred_x, pred_y = kalman_pred
        ky_x, ky_y = float(pred_x), float(np.squeeze(pred_x))
        ax, ay = acceleration[0], acceleration[1]

        features.extend([x, y, vx, vy, ax, ay, ky_x, ky_y])
        history_window = history[-WINDOWS_SIZE:]

        while len(history_window) < WINDOWS_SIZE:
            history_window.insert(0, history_window[0])

        for i in range(1, len(history_window)):
            dx = history_window[i].x - history_window[i - 1].x
            dy = history_window[i].y - history_window[i - 1].y
            features.extend([dx, dy])

        return np.array(features, dtype=np.float32)
    
    def save_ml_model(self):
        joblib.dump(self.ml_model, PLAYER_XGB_MODEL.as_posix())

    def handle_missed_players(self, frame_num: int):
        predicted_payloads = []
        if frame_num - value_store.globals.fps < 0:
            return []

        previous_states = context.analysis_context.tools.player_records.get_states_previous_frame(frame_num)

        if not previous_states:
            return []


        for state in previous_states:
            state = state.to_dict()
            player_id = state["player_id"]

            time_gap = frame_num - state["frame_index"]
            if time_gap > value_store.globals.fps * 2:
                continue

            pred_x, pred_y = self.predict_player_position(player_id)
            info_logger.info(f"[PlayerTracker] Kalman pred {pred_x}, {pred_y}")
            if pred_x is None or pred_y is None:
                continue

            pred_x = float(np.squeeze(pred_x))
            pred_y = float(np.squeeze(pred_y))

            payload = {
                "player_id": player_id,
                "frame_index": frame_num,
                "bbox": [pred_x - 30, pred_y - 45, pred_x + 30, pred_y + 45],
                "x": pred_x,
                "y": pred_y,
                "z": 0.0,
                "conf": 0.3,
                "timestamp_ms": value_store.globals.timestamp
            }

            predicted_payloads.append(payload)

        return predicted_payloads

    def validate_payloads(self, detected_players: list[dict], predicted_players: list[dict]):
        if len(predicted_players) == 0:
            return detected_players

        if len(detected_players) == 0:
            return predicted_players

        same_predicted_cords = [] # Keep ids from predicted players that has the same x and y fromm detected players
        threshold = self.movement_threshold * 0.5 # distance between both centers of the bbox to be considered the same player
        used_predicted = set()

        for detected_index, det in enumerate(detected_players):
            for predicted_index, pred in enumerate(predicted_players):
                dis = np.sqrt((det["x"] - pred["x"])**2 + (det["y"] - pred["y"])**2)
                if dis < threshold:
                    same_predicted_cords.append((detected_index, predicted_index))
        
        
        for detected_index, predicted_index in same_predicted_cords:
            detected_players[detected_index]["player_id"] = predicted_players[predicted_index]["player_id"]
            used_predicted.add(predicted_index)
        
        remaining_predicted = [
            pred for i, pred in enumerate(predicted_players)
            if i not in used_predicted
        ]

        # TODO Validar que sea correcto mezclar ambas listas
        info_logger.info(f"[PlayerTracker] Validando {len(detected_players)} players detectados y {len(remaining_predicted)} players predichos")
        info_logger.info(f"[PlayerTracker] Predicted: {remaining_predicted}")
        info_logger.info(f"[PlayerTracker] Detected: {detected_players}")
        
        return detected_players + remaining_predicted

    def predict_reid_probability(self, features: np.ndarray) -> float:
        try:
            x = features.reshape(1, -1)
            pred = self.ml_model.predict(x)
            
            if isinstance(pred, np.ndarray):
                return pred[0]
            else:
                return pred
            
        except Exception as e:
            info_logger.error(f"[PlayerTracker] Error al predecir reid: {e}")
            return 0
        