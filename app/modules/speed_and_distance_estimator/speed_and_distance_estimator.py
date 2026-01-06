from typing import Dict, List, Optional
import numpy as np
from scipy.signal import savgol_filter
from sqlalchemy.orm import Session

from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState
from app.entities.utils.global_values_store import GlobalValuesStore
from app.entities.utils.singleton import Singleton
from app.modules.services.bbox_processor_service import measure_scalar_distance
from app.logger import *

class SpeedAndDistanceEstimator(metaclass=Singleton):

    def __init__(
        self,
        frame_rate: float = 24,
        sprint_threshold_kmh: float = 25.0,
        smoothing_window: int = 7,
        poly_order: int = 2,
    ) -> None:
        self.globals = GlobalValuesStore()
        self.frame_rate = frame_rate
        self.sprint_threshold = sprint_threshold_kmh
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        self.max_acceleration_ms2 = self.globals.max_accel_ms2

    # ---------- helpers ----------
    def _smooth_positions(self, positions: List[np.ndarray]) -> np.ndarray:
        if len(positions) < self.smoothing_window:
            return positions[-1]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        try:
            xs_s = savgol_filter(xs, self.smoothing_window, self.poly_order, mode="nearest")
            ys_s = savgol_filter(ys, self.smoothing_window, self.poly_order, mode="nearest")
            return np.array([xs_s[-1], ys_s[-1]])
        except Exception:
            return positions[-1]

    # ---------- main entry ----------
    def process_track(
        self,
        frame_num: int,
        track_id: int,
        track: PlayerState,
        pixels_to_meters: float,
        camera_scale: float,
        dt: float,
        db: Session,
    ) -> None:
        try:
            info_logger.info(f"[SpeedAndDistance] Procesando track {track_id} en frame {frame_num}")

            x, y = float(f'{track.x}'), float(f'{track.y}')
            if x is None or y is None:
                info_logger.warning(f"[SpeedAndDistance] Posición inválida para track {track_id}")
                return

            pos = np.array([x, y])
            collection = TrackCollectionPlayer(db)
            states = collection.get_player_states(int(f'{track.player_id}'))

            # --- primer frame ---
            if len(states) < 2:
                collection.patch_state(
                    int(f'{track.player_id}'),
                    int(f'{track.frame_index}'),
                    {
                        "x_smoothed": x,
                        "y_smoothed": y,
                        "speed": 0.0,
                        "acceleration": 0.0,
                        "incremental_distance": 0.0,
                        "distance": 0.0,
                        "is_sprint": False,
                    },
                )
                return

            # --- suavizado ---
            valid_pos = [
                np.array([float(f'{s.x}'), float(f'{s.y}')])
                for s in states[-10:]
                if s.x is not None and s.y is not None
            ]
            if len(valid_pos) >= 2:
                smoothed_pos = self._smooth_positions(valid_pos)
            else:
                smoothed_pos = pos

            prev_state = states[-2]
            prev_pos = np.array([float(f'{prev_state.x}'), float(f'{prev_state.y}')])
            raw_dist_m = measure_scalar_distance(smoothed_pos, prev_pos) * pixels_to_meters / camera_scale

            # --- validaciones de rango ---
            if dt < self.globals.min_dt_s:
                info_logger.warning(
                    f"[SpeedAndDistance] dt muy pequeño ({dt:.4f}s) → se ignora frame"
                )
                return

            if raw_dist_m > self.globals.max_dist_per_frame_m:
                info_logger.warning(
                    f"[SpeedAndDistance] Salto de distancia inválido: {raw_dist_m:.2f}m → se fuerza 0m"
                )
                raw_dist_m = 0.0

            # --- velocidad ---
            speed_kmh = (raw_dist_m / dt) * 3.6
            if speed_kmh > self.globals.max_human_speed_kmh:
                # usar velocidad previa o media móvil
                last_speeds = [float(f'{s.speed}') for s in states[-5:] if s.speed is not None]
                if len(last_speeds) >= 2:
                    speed_kmh = float(np.mean(last_speeds))
                else:
                    speed_kmh = float(f'{prev_state.speed}') if prev_state.speed is not None else 0.0
                info_logger.warning(
                    f"[SpeedAndDistance] Velocidad fuera de rango corregida a {speed_kmh:.2f} km/h"
                )

            # --- aceleración ---
            prev_speed = float(f'{prev_state.speed}') if prev_state.speed is not None else 0.0
            acceleration_ms2 = (speed_kmh - prev_speed) / 3.6 / dt
            acceleration_ms2 = float(
                np.clip(acceleration_ms2, -self.max_acceleration_ms2, self.max_acceleration_ms2)
            )

            # --- distancia total ---
            total_distance = (float(f'{states[-1].distance}') if states[-1].distance is not None else 0.0) + raw_dist_m

            # --- sprint ---
            is_sprint = speed_kmh >= self.sprint_threshold

            # --- persistencia ---
            collection.patch_state(
                int(f'{track.player_id}'),
                int(f'{track.frame_index}'),
                {
                    "x_smoothed": float(smoothed_pos[0]),
                    "y_smoothed": float(smoothed_pos[1]),
                    "speed": speed_kmh,
                    "acceleration": acceleration_ms2,
                    "incremental_distance": raw_dist_m,
                    "distance": total_distance,
                    "is_sprint": is_sprint,
                },
            )

        except Exception as e:
            error_logger.error(f"[SpeedAndDistance] Error procesando track {track}: {e}")
            raise e
