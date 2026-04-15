import math
from typing import List
from functools import lru_cache
import numpy as np
from scipy.signal import savgol_filter
from sqlalchemy.orm import Session

from app.entities.models.PlayerModels import PlayerState
import app.entities.utils.tools_context as context
import app.entities.utils.global_values_store as value_store
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

        self.max_human_speed_kmh = value_store.globals.max_human_speed_kmh
        self.max_acceleration_ms2 = value_store.globals.max_accel_ms2
        self.max_dist_per_frame_m = value_store.globals.max_dist_per_frame_m
        self.min_dt_s = value_store.globals.min_dt_s

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
    
    def calculate_distance(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
        constant: float) -> float:
        dist_m = measure_scalar_distance(pos1, pos2)
        return float(dist_m)
    

    def process_track(
        self,
        frame_num: int,
        constant: float,
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
        players = context.analysis_context.tools.player_records.get_states_by_frame(frame_num)
        if not players or len(players) == 0:
            info_logger.info(f"[SpeedAndDistance] No se encuentran jugadores para el frame {frame_num}")
            return
        
        players = [player.to_dict() for player in players]

        for state in players:
            track_id = state["player_id"]
            frame_index = int(state['frame_index'])

            try:
                info_logger.info(
                    f"[SpeedAndDistance] Procesando track {track_id} en frame {frame_num}"
                )
                

                if frame_index == self.last_frame_calculated:
                    info_logger.info(f"[SpeedAndDistance] Track calculado anteriormente {track_id} en frame {frame_num}, ultimo frame calculado {self.last_frame_calculated}")
                    return
                x, y = float(state["x"]) * constant, float(state["y"]) * constant

                if not x or not y:
                    info_logger.warning(
                        f"[SpeedAndDistance] Posición inválida para track {track_id}"
                    )
                    return

                pos = np.array([x, y])

                prev_state = context.analysis_context.tools.player_records.get_previous_state(track_id, frame_num)
                info_logger.info(f"[SpeedAndDistance] Ulitmo frame calculado: {self.last_frame_calculated}")
                if prev_state is None:
                    info_logger.info(f"[SpeedAndDistance] No se encuentra el jugador {track_id} en el frame previo {frame_num}")
                    return
                
                prev_state = prev_state.to_dict()
                
                
                # if frame_num - 15 < self.last_frame_calculated:
                #     info_logger.info(f"[SpeedAndDistance] Ignorando track {track_id} en frame {frame_num}, ultimo frame calculado {self.last_frame_calculated}")
                #     return

                self.last_frame_calculated = frame_num

                states = context.analysis_context.tools.player_records.get_player_states(track_id)[:60]
                states = [s.to_dict() for s in states]
                if len(states) < 2:
                    info_logger.info(f"[SpeedAndDistance] No se encuentran suficientes registros para suavizar el track {track_id} en frame {frame_num}")
                    context.analysis_context.tools.player_records.patch_state(
                        track_id,
                        frame_index,
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
                    np.array([float(s["x"]), float(s["y"])])
                    for s in states
                    if s["x"] is not None and s["y"] is not None
                ]

                smoothed_pos = (
                    self._smooth_positions(valid_pos)
                    if len(valid_pos) >= self.smoothing_window // 2
                    else pos
                )

                # El tiempo esta en milisegundos se lo transforma a segundos
                time_o = float(prev_state['timestamp_ms']) * 0.001
                time_f = float(state["timestamp_ms"]) * 0.001
                dt = math.fabs(time_f - time_o)
                
                xf = (float(state['x']) * constant, float(state['y']) * constant)
                xo = (float(prev_state['x']) * constant, float(prev_state['y']) * constant)

                delta_x = np.array([xo[0] - xf[0], xo[1] - xf[1]])
                dist = math.fabs(np.linalg.norm(delta_x))
                vo = delta_x / dt
                acceleration = (2 * (delta_x - vo * dt)) / dt ** 2
                vf = vo + acceleration * dt
                speed_ms = np.linalg.norm(vf)
                speed_kmh = speed_ms * 3.6
                acceleration_ms2 = np.linalg.norm(acceleration)
                
                is_sprint = False
                if speed_kmh > self.max_human_speed_kmh or speed_kmh > self.sprint_threshold:
                    speed_kmh = self.max_human_speed_kmh
                    is_sprint = True

                total_distance = context.analysis_context.tools.player_records.calculate_player_total_distance(track_id) + dist
                info_logger.info(f"[SpeedAndDistance] Distancia calculada para {track_id}: {dist} m, distancia: {total_distance} m, dist calc {dist}, velocidad: {speed_kmh} km/h")

                context.analysis_context.tools.player_records.patch_state(
                    track_id,
                    frame_index,
                    {
                        "x_smoothed": float(smoothed_pos[0]),
                        "y_smoothed": float(smoothed_pos[1]),
                        "speed": speed_kmh,
                        "acceleration": acceleration_ms2,
                        "incremental_distance": total_distance,
                        "distance": dist,
                        "is_sprint": is_sprint,
                    },
                )
                context.analysis_context.tools.analysis_data_collector.update_row(
                    frame=frame_num,
                    track_id=track_id,
                    velocity=float(speed_kmh),
                )
                states = None
            except Exception as e:
                error_logger.error(
                    f"[SpeedAndDistance] Error procesando velocidad y distancia del track {state}: {e}"
                )
                raise e
