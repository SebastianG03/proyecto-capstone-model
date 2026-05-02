import math
from typing import List, Tuple

import joblib
import numpy as np

from app.entities.models.PlayerModels import PlayerState, State, PlayerStatus
import app.entities.utils.global_values_store as value_store
from app.entities.trackers.player.player_kalman_filter import PlayerKalmanFilter
import app.entities.utils.tools_context as context
from app.logger import info_logger
import xgboost as xgb

from app.utils.routes import PLAYER_XGB_MODEL

class PlayerMatcher():
    def __init__(self,):
        self.movement_threshold = 15
        self.kalman_filters: dict[int, PlayerKalmanFilter] = {}
        self._training_buffer: list[tuple[np.ndarray, int]] = []
        self.BUFFER_MIN_SAMPLES = 500
        self.NEGATIVE_SAMPLES_PER_MATCH = 2
        self.last_timestamps: dict[int, float] = {}
        self.CHI_GATE = np.sqrt(5.991)
        self.id_switch_votes = {}   
        self.player_missed_counts: dict[int, int] = {} # track_id -> missed_count
        self.MAX_SPEED = 30
        self.MAX_MISSED_PLAYERS = 10
        self.MAX_CONF = .25

        self.player_model_exists = (PLAYER_XGB_MODEL / "player_model.pkl").exists()

        if self.player_model_exists:
            self.ml_model = joblib.load((PLAYER_XGB_MODEL / "player_model.pkl").as_posix())
            self.ml_model.set_params(n_estimators=200, max_depth=6, learning_rate=0.05)
        else:
            self.ml_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                objective="binary:logistic"
            )


    @staticmethod
    def filter_states(states: List[PlayerState], column_name: str, reverse: bool = False) -> List[dict]:        
        if states is None or len(states) == 0:
            return []
        
        dict_states = [state.to_dict() for state in states]
        dict_states.sort(key=lambda s: s[column_name], reverse=reverse)

        return dict_states
    
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

    def compute_match_cost(
        self,
        player_id: int,
        payload: dict,
        prev_state: dict,
        kf: PlayerKalmanFilter
    ):
        payload_x, payload_y = float(payload["x"]), float(payload["y"])
        maha_dist = kf.mahalanobis_distance(payload_x, payload_y)
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
            .5 * (maha_dist / self.CHI_GATE) +
            .3 * (dist / self.movement_threshold) +
            .2 * (residual / self.movement_threshold)
        )
        
    def _get_or_create_kf(self, player_id: int, x: float, y: float) -> PlayerKalmanFilter:
        if player_id not in self.kalman_filters:
            kf = PlayerKalmanFilter(dt=1/30)
            kf.x[:2] = np.array([[x], [y]])
            self.kalman_filters[player_id] = kf
        else:
            kf = self.kalman_filters[player_id]
        
        return kf

    def generate_candidates(self, payload: dict, previous_states: List[dict], kf: PlayerKalmanFilter):
        candidates = []
        
        x = float(payload["x"])
        y = float(payload["y"])
        
        for state in previous_states:
            player_id = state["player_id"]
            state_x, state_y = float(state["x"]), float(state["y"])
            maha_dist = kf.mahalanobis_distance(x, y)

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

            candidates.append((player_id, maha_dist, dist))
            
        return candidates
    
    def _build_cost_matrix(self, detections: List[dict], previous_states: List[dict]):
        N, M = len(detections), len(previous_states)
        matrix = np.full((N, M), np.inf)
        
        for i, det in enumerate(detections):
            for j, state in enumerate(previous_states):
                player_id = state["player_id"]
                kf = self._get_or_create_kf(player_id, float(det["x"]), float(det["y"]))
                cost = self.compute_match_cost(player_id, det, state, kf)
                features = self.build_match_features(state, det, kf)
                ml_score = self._ml_score(features)
                if cost is not None:
                    final_score = (1 - ml_score) * cost
                    matrix[i][j] = final_score

        return matrix
    
    def _ml_score(self, features: np.ndarray):
        if not self.player_model_exists:
            return 0.5
        try:
            prob = self.ml_model.predict_proba(features.reshape(1, -1))[0][1]
            return float(prob)
        except Exception as e:
            info_logger.warning(f"[PlayerMatcher] ML predict_proba falló: {e}")
            return 0.5

    def fit_ml_model(self, X: np.ndarray, y: np.ndarray):
        """Entrena el modelo con datos acumulados. Llama esto offline."""
        self.ml_model.fit(X, y)
        self._ml_ready = True
        self.save_ml_model()
        info_logger.info(f"[PlayerMatcher] Modelo entrenado con {len(X)} muestras.")

    def validate_previous_players(self, frame_num, state_payloads: List[dict]):
        if frame_num <= value_store.globals.fps * 2:
            info_logger.info(f"[PlayerTracker] Insuficientes frames para validar estados anteriores en frame {frame_num}")
            return state_payloads

        prev_states = context.analysis_context.tools.player_records.get_states_previous_frame(frame_num)
        previous_states = PlayerMatcher.filter_states(prev_states, "frame_index", True)
        
        if len(previous_states) == 0:
            return state_payloads

        filtered_states = self.get_unique_previous_states(previous_states, frame_num)
        cost_matrix = self._build_cost_matrix(state_payloads, filtered_states)
        assigned_det = set()
        assigned_prev = set()
        
        pairs = [
            (cost_matrix[i, j], i, j)
            for i in range(len(state_payloads))
            for j in range(len(filtered_states))
            if not np.isinf(cost_matrix[i, j])
        ]
        pairs.sort(key=lambda x: x[0])
        
        for cost, det_idx, prev_idx in pairs:
            if det_idx in assigned_det or prev_idx in assigned_prev:
                continue
            
            prev_id = int(filtered_states[prev_idx]["player_id"])
            current_id = int(state_payloads[det_idx]["player_id"])
            kf = self._get_or_create_kf(prev_id, float(state_payloads[det_idx]["x"]), float(state_payloads[det_idx]["y"]))
            
            self._collect_training_sample(
                state_payloads[det_idx],
                filtered_states[prev_idx],
                filtered_states,
                prev_idx,
                kf
            )

            if current_id != prev_id:
                info_logger.info(f"[PlayerTracker] Player_id {current_id} en frame {frame_num} no coincide con player_id {prev_id} del frame anterior con distancia {cost:.4f}")
                state_payloads[det_idx]["player_id"] = prev_id
            
            assigned_det.add(det_idx)
            assigned_prev.add(prev_idx)

        for j in range(len(filtered_states)):
            if j not in assigned_prev:
                prev_id = int(filtered_states[j]["player_id"])
                info_logger.info(f"[PlayerTracker] Player_id {prev_id} del frame anterior no asignado en frame {frame_num}")
                self.player_missed_counts[prev_id] += 1
        

        detected_ids = {p["player_id"] for p in state_payloads}
        predicted_payloads = self.handle_missed_players(frame_num=frame_num, detected_ids=detected_ids)
        validated_payloads = self.validate_payloads(detected_players=state_payloads, predicted_players=predicted_payloads)

        return validated_payloads

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
        
    def build_match_features(self, prev_state: dict, candidate: dict, kf: PlayerKalmanFilter):
        px, py = prev_state["x"], prev_state["y"]
        cx, cy = candidate["x"], candidate["y"]
        timestamp = math.fabs((float(candidate["timestamp_ms"]) - float(prev_state["timestamp_ms"])) * 0.001) 
        
        maha = kf.mahalanobis_distance(cx, cy)
        dist = np.linalg.norm(np.array([cx, cy]) - np.array([px, py]))
        
        pred_x, pred_y = self.predict_player_position(prev_state["player_id"])
        
        if pred_x is None:
            residual = dist
        else:
            residual = np.linalg.norm(np.array([pred_x, pred_y]) - np.array([cx, cy]))
        
        return np.array([
            maha, dist, residual, timestamp,
            self.player_missed_counts.get(prev_state["player_id"], 0)
        ], dtype=np.float32)

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

        z = np.array([state.x, state.y])
        kf.update(z, dt)

        self.last_timestamps[player_id] = current_time

        return kf.x

    def predict_player_position(self, player_id: int):
        if player_id not in self.kalman_filters:
            info_logger.info(f"[PlayerTracker] No hay filtro de Kalman para player_id {player_id}, no se puede predecir posicion.")  
            return None, None

        kf = self.kalman_filters[player_id]
        x_pred, P_pred = kf.predict()
        info_logger.info(f"[PlayerTracker] Kalman pred {x_pred}, {P_pred}")
        
        # pred_x = float(x_pred[0][0])
        # pred_y = float(x_pred[1][0])

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
        vx, vy = state.vx, state.vy
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

    def handle_missed_players(self, frame_num: int, detected_ids: set[int]):
        predicted_payloads = []
        if frame_num < value_store.globals.fps:
            return []

        previous_states = context.analysis_context.tools.player_records.get_states_previous_frame(frame_num)
        if not previous_states:
            return []

        for state in previous_states:
            state = state.to_dict()
            player_id = state["player_id"]

            if player_id in detected_ids:
                continue

            missed = self.player_missed_counts.get(player_id, 0)
            
            if missed >= self.MAX_MISSED_PLAYERS:
                info_logger.info(f"[PlayerTracker] Jugador {player_id} perdido durante {missed} frames, se omite.")
                continue

            time_gap = frame_num - state["frame_index"]
            if time_gap > value_store.globals.fps * 2:
                continue

            pred_x, pred_y = self.predict_player_position(player_id)
            info_logger.info(f"[PlayerTracker] Kalman pred {pred_x}, {pred_y}")
            if pred_x is None or pred_y is None:
                continue

            pred_x = float(np.squeeze(pred_x))
            pred_y = float(np.squeeze(pred_y))
            
            conf = max(0.1, self.MAX_CONF - missed * 0.02)

            payload = {
                "player_id": player_id,
                "frame_index": frame_num,
                "bbox": [pred_x - 30, pred_y - 45, pred_x + 30, pred_y + 45],
                "x": pred_x,
                "y": pred_y,
                "z": 0.0,
                "conf": conf,
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
        threshold = self.movement_threshold * 2 # distance between both centers of the bbox to be considered the same player
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

        info_logger.info(f"[PlayerTracker] Validando {len(detected_players)} players detectados y {len(remaining_predicted)} players predichos")
        info_logger.info(f"[PlayerTracker] Predicted: {remaining_predicted}")
        info_logger.info(f"[PlayerTracker] Detected: {detected_players}")
        
        return detected_players + remaining_predicted

    def _collect_training_sample(
    self,
    confirmed_det: dict,
    confirmed_prev: dict,
    all_prev_states: list[dict],
    confirmed_prev_idx: int,
    kf: PlayerKalmanFilter,
):
        # Positivo: el par que Hungarian confirmó como correcto
        features_pos = self.build_match_features(confirmed_prev, confirmed_det, kf)
        self._training_buffer.append((features_pos, 1))

        # Negativos: mismo det contra otros estados previos (falsos candidatos)
        negatives_added = 0
        for j, other_prev in enumerate(all_prev_states):
            if j == confirmed_prev_idx:
                continue
            if negatives_added >= self.NEGATIVE_SAMPLES_PER_MATCH:
                break

            other_id = other_prev["player_id"]
            other_kf = self.kalman_filters.get(other_id)
            if other_kf is None:
                continue

            features_neg = self.build_match_features(other_prev, confirmed_det, other_kf)
            self._training_buffer.append((features_neg, 0))
            negatives_added += 1

    def train_from_buffer(self) -> bool:
        total = len(self._training_buffer)
        if total < self.BUFFER_MIN_SAMPLES:
            info_logger.info(
                f"[PlayerMatcher] Buffer insuficiente para entrenar: "
                f"{total}/{self.BUFFER_MIN_SAMPLES} muestras"
            )
            return False

        X = np.array([f for f, _ in self._training_buffer], dtype=np.float32)
        y = np.array([l for _, l in self._training_buffer], dtype=np.int32)

        pos = int(y.sum())
        neg = int((y == 0).sum())
        info_logger.info(f"[PlayerMatcher] Entrenando con {pos} positivos y {neg} negativos")

        self.fit_ml_model(X, y)
        self._training_buffer.clear()
        return True