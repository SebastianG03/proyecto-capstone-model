from typing import List
from functools import lru_cache
import numpy as np
from scipy.signal import savgol_filter
from sqlalchemy.orm import Session

from app.entities.collections import TrackCollectionPlayer, player_collections
from app.entities.models.PlayerModels import PlayerState
from app.entities.utils.global_values_store import globals
from app.entities.utils.singleton import Singleton
from app.infraestructure.services.bbox_processor_service import measure_scalar_distance
from app.logger import info_logger, error_logger


class SpeedAndDistanceEstimator(metaclass=Singleton):
    def __init__(
        self,
        frame_rate: float = 24.0,
        sprint_threshold_kmh: float = 25.0,
        smoothing_window: int = 7,
        poly_order: int = 2,
    ) -> None:
        """
        Inicializa el estimador de velocidad y distancia.

        Args:
            frame_rate (float): Frecuencia de frames por segundo (por defecto 24.0).
            sprint_threshold_kmh (float): Umbral de velocidad en km/h
            para considerar un sprint (por defecto 25.0).
            smoothing_window (int): Tamaño de la ventana para suavizar
            las posiciones de los jugadores (por defecto 7).
            poly_order (int): Orden de la ecuación de Savitzky-Golay para suavizar las posiciones
            de los jugadores (por defecto 2).
        """
        self.frame_rate = frame_rate
        self.sprint_threshold = sprint_threshold_kmh
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        self.last_frame_calculated = 0

        self.max_human_speed_kmh = globals.max_human_speed_kmh
        self.max_acceleration_ms2 = globals.max_accel_ms2
        self.max_dist_per_frame_m = globals.max_dist_per_frame_m
        self.min_dt_s = globals.min_dt_s

    def _smooth_positions(self, positions: List[np.ndarray]) -> np.ndarray:
        """
        Suaviza las posiciones de los jugadores aplicando una ventana de Savitzky-Golay.
        Si no se puede aplicar la suavización (n < self.smoothing_window),
        se devuelve la última posición.
        """
        n = len(positions)
        if n < self.smoothing_window:
            return positions[-1]

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        try:
            xs_s = savgol_filter(
                xs, self.smoothing_window, self.poly_order, mode="nearest"
            )
            ys_s = savgol_filter(
                ys, self.smoothing_window, self.poly_order, mode="nearest"
            )
            return np.array([xs_s[-1], ys_s[-1]])
        except Exception:
            return positions[-1]

    @lru_cache(maxsize=128)
    def _get_states(self, player_id: int, db: Session) -> List[PlayerState]:
        """
        Obtiene una lista de registros de un jugador en la base de datos.

        Args:
            player_id (int): Identificador del usuario.
            db (Session): Sesión de la base de datos.

        Returns:
            List[PlayerState]: Lista de registros de un usuario en la base de datos.
        """
        return TrackCollectionPlayer(db).get_player_states(player_id)[:20]
    
    def calculate_distance(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
        pixels_to_meters: float,
        depth: float = 1,
        camera_scale: float = 1,
        ) -> float:
        dist_px = measure_scalar_distance(pos1, pos2)
        dist_m = dist_px * pixels_to_meters * depth / camera_scale
        return float(dist_m)
    
    def get_previous_state(self, players: List[PlayerState], actual_frame: int):
        actual_frame = 0
        last_recognized_pos = len(players) // 3
        previous_state = players[last_recognized_pos]

        for player in players:
            frame_index = int(f'{player.frame_index}')
            
            if frame_index % 15 == 0 or actual_frame - 15 >  self.last_frame_calculated:
                previous_state = player
                break

        return previous_state

    def process_track(
        self,
        frame_num: int,
        track_id: int,
        track: PlayerState,
        pixels_to_meters: float,
        camera_scale: float,
        depth: float,
        dt: float,
        db: Session,
    ) -> None:
        """
        Procesa un track de un jugador y calcula velocidad, aceleración y distancia total.
        Se encarga de suavizar las posiciones del usuario, calcular la velocidad y aceleración,
        y de validar los resultados para asegurar que se encuentren dentro de los rangos
        de velocidad y aceleración permitidos. También se encarga de persistir los resultados
        en la base de datos.
        """

        try:
            info_logger.info(
                f"[SpeedAndDistance] Procesando track {track_id} en frame {frame_num}"
            )
            
            x, y = float(f"{track.x}"), float(f"{track.y}")
            pos = np.array([x, y])
            pid = int(float(f"{track.player_id}"))
            fidx = int(float(f"{track.frame_index}"))
            
            if frame_num - 15 < self.last_frame_calculated:
                return
            
            if fidx == self.last_frame_calculated:
                return
            
            self.last_frame_calculated = frame_num

            if x is None or y is None:
                info_logger.warning(
                    f"[SpeedAndDistance] Posición inválida para track {track_id}"
                )
                return

            states = self._get_states(pid, db)
            if len(states) < 2:
                TrackCollectionPlayer(db).patch_state(
                    pid,
                    fidx,
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

            valid_pos = [
                np.array([float(f"{s.x}"), float(f"{s.y}")])
                for s in states[:12]
                if s.x is not None and s.y is not None
            ]
            smoothed_pos = (
                self._smooth_positions(valid_pos)
                if len(valid_pos) >= self.smoothing_window // 2
                else pos
            )

            prev_state = self.get_previous_state(states, frame_num)
            prev_pos = np.array([float(f'{prev_state.x}'), float(f'{prev_state.y}')])
            raw_dist_m = self.calculate_distance(
                smoothed_pos,
                prev_pos,
                pixels_to_meters,
                depth,
                camera_scale)

            if dt <= 0.0 or dt < self.min_dt_s:
                info_logger.warning(
                    f"[SpeedAndDistance] dt muy pequeño ({dt:.4f}s) → se ignora frame"
                )
                raw_dist_m = 0.5

            if raw_dist_m > self.max_dist_per_frame_m:
                info_logger.warning(
                    f"[SpeedAndDistance] Salto de distancia inválido: {raw_dist_m:.2f}m "
                    "→ se fuerza 0m"
                )
                raw_dist_m = 5.0

            speed_kmh = (raw_dist_m / dt) * 3.6
            if speed_kmh > self.max_human_speed_kmh:
                last_speeds = [
                    float(f"{s.speed}")
                    for s in states[-10:]
                    if s.speed is not None
                    and float(f"{s.speed}") <= self.max_human_speed_kmh
                ]
                speed_kmh = float(np.median(last_speeds)) if last_speeds else 0.0
                info_logger.warning(
                    f"[SpeedAndDistance] Velocidad fuera de rango corregida a {speed_kmh:.2f} km/h"
                )

            prev_speed = (
                float(f"{prev_state.speed}") if prev_state.speed is not None else 0.0
            )

            acceleration_ms2 = (speed_kmh - prev_speed) / 3.6 / dt
            acceleration_ms2 = float(
                np.clip(
                    acceleration_ms2,
                    -self.max_acceleration_ms2,
                    self.max_acceleration_ms2,
                )
            )

            total_distance = TrackCollectionPlayer(db).calculate_player_total_distance(pid)

            is_sprint = speed_kmh >= self.sprint_threshold

            TrackCollectionPlayer(db).patch_state(
                pid,
                fidx,
                {
                    "x_smoothed": float(smoothed_pos[0]),
                    "y_smoothed": float(smoothed_pos[1]),
                    "speed": speed_kmh,
                    "acceleration": acceleration_ms2,
                    "incremental_distance": speed_kmh / 3.6 * dt,
                    "distance": total_distance,
                    "is_sprint": is_sprint,
                },
            )
            states = None
        except Exception as e:
            error_logger.error(
                f"[SpeedAndDistance] Error procesando track {track}: {e}"
            )
            raise e
