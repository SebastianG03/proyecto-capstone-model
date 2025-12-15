from typing import Dict, List, Optional
import numpy as np
from scipy.signal import savgol_filter
from sqlalchemy.orm import Session

from app.entities.collections.track_collections import TrackCollectionPlayer
from app.entities.models.PlayerState import PlayerStateModel
from app.entities.utils.singleton import Singleton
from app.modules.services.bbox_processor_service import measure_scalar_distance
from app.logger import *


class SpeedAndDistanceEstimator(metaclass=Singleton):

    def __init__(
        self,
        frame_rate: int = 24,
        sprint_threshold_kmh: float = 25.0,
        smoothing_window: int = 7,
        poly_order: int = 2,
        history_size: int = 60,
    ) -> None:

        self.frame_rate = frame_rate
        self.sprint_threshold = sprint_threshold_kmh
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        self.history_size = history_size

        self.position_history: Dict[int, List[Optional[np.ndarray]]] = {}
        self.speed_history: Dict[int, List[float]] = {}

    # ----------------------------
    # Utilidades internas
    # ----------------------------

    def _smooth_values(self, values: List[float]) -> List[float]:
        if len(values) < self.smoothing_window:
            return values
        try:
            return list(
                savgol_filter(values, self.smoothing_window, self.poly_order, mode="nearest")
            )
        except ValueError:
            return values

    def _interpolate_last(self, values: List[Optional[np.ndarray]]) -> np.ndarray:
        if len(values) < 3:
            return np.array([0.0, 0.0])
        v1, v2 = values[-2], values[-3]
        if v1 is None or v2 is None:
            return np.array([0.0, 0.0])
        return v1 + (v1 - v2)

    def _smooth_positions_array(self, positions: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """
        Smooths an array of positions (x, y) using Savitzky-Golay filter.

        Args:
            positions (List[Optional[np.ndarray]]): List of positions (x, y)

        Returns:
            List[np.ndarray]: Smooothed list of positions (x, y)
        """

        xs = [p[0] for p in positions if p is not None]
        ys = [p[1] for p in positions if p is not None]

        xs_smooth = self._smooth_values(xs)
        ys_smooth = self._smooth_values(ys)

        return [np.array([x, y]) for x, y in zip(xs_smooth, ys_smooth)]

    # ----------------------------
    # PROCESAMIENTO DE TRACKS
    # ----------------------------

    def process_track(
        self,
        frame_num: int,
        track_id: int,
        track: PlayerStateModel,
        pixels_to_meters: float,
        camera_scale: float,
        db: Session,
    ) -> None:
        """
        Procesa un track de un jugador en un frame.

        Args:
            frame_num (int): Número de frame relativo al batch/frame procesado
            track_id (int): Identificador del jugador
            track (PlayerStateModel): Estado del usuario en el frame
            db (Session): Sesión de base de datos
            model_class: Clase de modelo para persistir resultados

        Returns:
            None
        """
        try:
            info_logger.info(f"[SpeedAndDistance] Procesando distancia y velocidad para track {track_id} en frame {frame_num}")
            x = track.x
            y = track.y 
            info_logger.info(f"[SpeedAndDistance] Posición del track para velocidad y distancia {track_id} en frame {frame_num}: x={x}, y={y}")
            if x is None or y is None:
                info_logger.info(f"[SpeedAndDistance] No hay posición para track {track_id} en frame {frame_num}, saltando procesamiento.")
                return
            pos = np.array([x, y])  # Debe ser np.array([x,y]) después de homografía
            info_logger.info(f"[SpeedAndDistance] Procesando velocidad y posicion para track {track_id} en frame {frame_num} con posición {pos}")
            # Inicializar buffers
            if track_id not in self.position_history:
                info_logger.info(f"[SpeedAndDistance] Inicializando buffers para track velocidad y posicion {track_id}")
                self.position_history[track_id] = []
            if track_id not in self.speed_history:
                info_logger.info(f"[SpeedAndDistance] Inicializando buffers para track velocidad y posicion {track_id}")
                self.speed_history[track_id] = []

            # Agregar posición
            self.position_history[track_id].append(pos)
            debug_logger.debug(f"[SpeedAndDistance] Posición agregada para track {track_id}: {pos}")
            # Mantener tamaño del buffer
            if len(self.position_history[track_id]) > self.history_size:
                info_logger.info(f"[SpeedAndDistance] Manteniendo tamaño del buffer para track {track_id}")
                self.position_history[track_id].pop(0)
            info_logger.info(f"[SpeedAndDistance] Historial de posiciones para track {track_id}: {self.position_history[track_id]}")
            # Interpolación si no hay detección
            if pos is None:
                info_logger.info(f"[SpeedAndDistance] No hay detección para track {track_id}")
                interpolated = self._interpolate_last(self.position_history[track_id])
                info_logger.info(f"[SpeedAndDistance] Posición interpolada para track {track_id}: {interpolated}")
                self.position_history[track_id][-1] = interpolated
                pos = interpolated

            # Suavizado
            smooth_positions = self._smooth_positions_array(self.position_history[track_id])
            print(f"[SpeedAndDistance] Smoothed position for track id {track_id} on frame num {frame_num} positions are: {smooth_positions}")
            debug_logger.debug(f"[SpeedAndDistance] Posiciones suavizadas para track {track_id}: {smooth_positions}")
            smoothed_pos = smooth_positions[-1]
            debug_logger.debug(f"Posición suavizada para track {track_id}: {smoothed_pos}")

            # Velocidad
            if len(smooth_positions) >= 2:
                # Distancia en metros
                dist_m = measure_scalar_distance(smooth_positions[-1], smooth_positions[-2]) * pixels_to_meters
                debug_logger.debug(f"[SpeedAndDistance] Distancia calculada para track {track_id}: {dist_m} metros")
                debug_logger.debug(f"[SpeedAndDistance] Distancia calculada para track con escala {track_id}: {dist_m / camera_scale} metros")
                debug_logger.debug(f"[SpeedAndDistance] Distancia calculada para track con escala multiplicada {track_id}: {dist_m * camera_scale} metros")
                speed_kmh = (dist_m * self.frame_rate) * 3.6
                debug_logger.debug(f"[SpeedAndDistance] Velocidad calculada para track {track_id}: {speed_kmh} km/h")
            else:
                speed_kmh = 0.0

            # Guardar velocidad histórica
            self.speed_history[track_id].append(speed_kmh)
            if len(self.speed_history[track_id]) > self.history_size:
                info_logger.info(f"[SpeedAndDistance] Manteniendo tamaño del buffer de velocidad para track {track_id}")
                self.speed_history[track_id].pop(0)
            debug_logger.debug(f"[SpeedAndDistance] Historial de velocidades para track {track_id}: {self.speed_history[track_id]}")

            smooth_speed_kmh = self._smooth_values(self.speed_history[track_id])[-1]
            debug_logger.debug(f"[SpeedAndDistance] Velocidad suavizada para track {track_id}: {smooth_speed_kmh} km/h vs calculada {speed_kmh} km/h")

            # Aceleración
            if len(self.speed_history[track_id]) >= 2:
                v1, v2 = self.speed_history[track_id][-1], self.speed_history[track_id][-2]
                acceleration = (v1 - v2) / (1 / self.frame_rate)
            else:
                acceleration = 0.0
            debug_logger.debug(f"[SpeedAndDistance] Aceleración calculada para track {track_id}: {acceleration} km/h²")

            # Distancia incremental
            if len(smooth_positions) >= 2:
                incremental_dist = measure_scalar_distance(smooth_positions[-1], smooth_positions[-2])
                debug_logger.debug(f"[SpeedAndDistance] Distancia incremental para track {track_id}: {incremental_dist} metros")
            else:
                incremental_dist = 0.0
            debug_logger.debug(f"[SpeedAndDistance] Distancia incremental para track {track_id}: {incremental_dist} metros")

            # Distancia total
            total_distance = float(sum(
                measure_scalar_distance(p1, p2)
                for p1, p2 in zip(smooth_positions[:-1], smooth_positions[1:])
            ))
            debug_logger.debug(f"[SpeedAndDistance] Distancia total para track {track_id}: {total_distance} metros")

            # Sprint
            is_sprint = smooth_speed_kmh >= self.sprint_threshold
            debug_logger.debug(f"[SpeedAndDistance] ¿Está sprintando el track {track_id}? {'Sí' if is_sprint else 'No'}")
            # Persistencia
            data_updated = {
                "player_id": track_id,
                "frame_index": frame_num,
                "x_smoothed": float(smoothed_pos[0]),
                "y_smoothed": float(smoothed_pos[1]),
                "speed": float(smooth_speed_kmh),
                "acceleration": float(acceleration),
                "incremental_distance": float(incremental_dist),
                "distance": float(total_distance),
                "is_sprint": is_sprint,
            }

            player_collection = TrackCollectionPlayer(db)
            res = player_collection.patch(int(f'{track.id}'), data_updated)
            debug_logger.debug(f"[SpeedAndDistance] Track {track_id} actualizado en DB: {res}")
        except Exception as e:
            error_logger.error(f"[SpeedAndDistance] Error procesando track {track}: {e}")
            raise e