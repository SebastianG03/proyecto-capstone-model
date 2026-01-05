from typing import Dict, List, Optional
import numpy as np
from scipy.signal import savgol_filter
from sqlalchemy.orm import Session

from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState
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
        max_acceleration_ms2: float = 4.0,  # límite realista para amateur
    ) -> None:
        self.frame_rate = frame_rate
        self.sprint_threshold = sprint_threshold_kmh
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        self.max_acceleration_ms2 = max_acceleration_ms2

    # ------------------------------
    # Utilidades internas
    # ------------------------------

    def _smooth_positions(self, positions: List[np.ndarray]) -> np.ndarray:
        if len(positions) < self.smoothing_window:
            return positions[-1]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        try:
            xs_smooth = savgol_filter(xs, self.smoothing_window, self.poly_order, mode="nearest")
            ys_smooth = savgol_filter(ys, self.smoothing_window, self.poly_order, mode="nearest")
            return np.array([xs_smooth[-1], ys_smooth[-1]])
        except Exception:
            return positions[-1]

    # ------------------------------
    # Procesamiento de tracks
    # ------------------------------

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

            x = float(f'{track.x}')
            y = float(f'{track.y}')
            if x is None or y is None:
                info_logger.warning(f"[SpeedAndDistance] Posición inválida para track {track_id}")
                return

            pos = np.array([x, y])

            # Obtener estados anteriores del jugador
            collection = TrackCollectionPlayer(db)
            states = collection.get_player_states(int(f'{track.player_id}'))
            if len(states) < 2:
                # Primera aparición o insuficiente
                data_updated = {
                    "x_smoothed": x,
                    "y_smoothed": y,
                    "speed": 0.0,
                    "acceleration": 0.0,
                    "incremental_distance": 0.0,
                    "distance": 0.0,
                    "is_sprint": False,
                }
                collection.patch_state(int(f'{track.player_id}'), int(f'{track.frame_index}'), data_updated)
                return

            # Filtrar posiciones válidas
            valid_positions = []
            for s in states[-10:]:  # últimos 10 estados
                if s.x is not None and s.y is not None:
                    valid_positions.append(np.array([float(f'{s.x}'), float(f'{s.y}')]))

            if len(valid_positions) < 2:
                smoothed_pos = pos
            else:
                smoothed_pos = self._smooth_positions(valid_positions)

            # Distancia entre últimas 2 posiciones
            prev_pos = np.array([float(f'{states[-2].x}'), float(f'{states[-2].y}')])
            raw_distance_m = measure_scalar_distance(smoothed_pos, prev_pos) * pixels_to_meters / camera_scale

            # Limitar salto a 1 km (por pérdida de tracking)
            if raw_distance_m > 1000:
                info_logger.warning(f"[SpeedAndDistance] Salto de distancia inválido: {raw_distance_m:.2f}m → 0m")
                raw_distance_m = 0.0

            # Velocidad en km/h
            speed_kmh = (raw_distance_m / dt) * 3.6

            # Promediar si velocidad es muy alta (> 30 km/h)
            if speed_kmh > 30.0:
                last_speeds = [float(f'{s.speed}') for s in states[-5:] if s.speed is not None]
                if len(last_speeds) >= 3:
                    avg_speed = np.mean(last_speeds)
                    speed_kmh = (speed_kmh + avg_speed) / 2

            # Aceleración (m/s²)
            prev_speed = float(f'{states[-2].speed}') if states[-2].speed is not None else 0.0
            acceleration_ms2 = (speed_kmh - prev_speed) / 3.6 / dt  # km/h → m/s
            acceleration_ms2 = np.clip(acceleration_ms2, -self.max_acceleration_ms2, self.max_acceleration_ms2)

            # Distancia total acumulada
            total_distance = (float(f'{states[-1].distance}') if states[-1].distance is not None else 0.0) + raw_distance_m

            # ¿Sprint?
            is_sprint = speed_kmh >= self.sprint_threshold

            # Persistencia
            data_updated = {
                "x_smoothed": float(smoothed_pos[0]),
                "y_smoothed": float(smoothed_pos[1]),
                "speed": float(speed_kmh),
                "acceleration": float(acceleration_ms2),
                "incremental_distance": float(raw_distance_m),
                "distance": float(total_distance),
                "is_sprint": is_sprint,
            }

            collection.patch_state(int(f'{track.player_id}'), int(f'{track.frame_index}'), data_updated)

        except Exception as e:
            error_logger.error(f"[SpeedAndDistance] Error procesando track {track}: {e}")
            raise e
