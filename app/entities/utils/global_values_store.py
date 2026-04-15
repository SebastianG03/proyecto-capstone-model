from typing import List, Optional

from sqlalchemy.orm import Session

import app.entities.utils.singleton as singleton
import app.entities.models.detected_object_data as detected_object_data
import app.infraestructure.database.connection_manager as connection
import app.entities.services.video_anotator as anotator


class GlobalValuesStore(metaclass=singleton.Singleton):
    """
    Almacena y permite actualizar dos valores: timestamp (float) y fps (float).
    """

    def __init__(self, timestamp: float = 0.0, fps: float = 24.0) -> None:
        """
        Inicializa la clase GlobalValuesStore.

        Args:
            timestamp (float): Valor del timestamp a inicializar (por defecto 0.0).
            fps (float): Valor de los fps a inicializar (por defecto 0.0).
        """
        self._timestamp: float = float(timestamp)
        self._fps: float = float(fps)
        self.MAX_HUMAN_SPEED_KMH = 30.0
        self.MAX_ACCEL_MS2 = 6.0
        self.MAX_DIST_PER_FRAME_M = self.MAX_HUMAN_SPEED_KMH / 3.6 / 24  # fps
        self.MIN_DT_S = 1.0 / (2 * 24)
        self._connection_manager: connection.ConnectionManager
        self._session: Session
        self._frame_size = (1280, 720)
        self.anotated_colors = {
            "player": (56,226,235),
            "ball": (235,79,22),
            "goal": (19,102,12)
        }
        self._depth = 1
        self._video_anotator: anotator.VideoAnotator
        self._detected_object: List[detected_object_data.DetectedObjectData] = []

    # --- Getters ---
    @property
    def timestamp(self) -> float:
        """Devuelve el valor actual del timestamp."""
        return self._timestamp

    @property
    def fps(self) -> float:
        """Devuelve el valor actual de los fps."""
        return self._fps

    @property
    def max_human_speed_kmh(self) -> float:
        """Devuelve el valor actual de la velocidad máxima de un humano en km/h."""
        return self.MAX_HUMAN_SPEED_KMH

    @property
    def max_accel_ms2(self) -> float:
        """Devuelve el valor actual de la aceleración máxima de un humano en m/s^2."""
        return self.MAX_ACCEL_MS2

    @property
    def max_dist_per_frame_m(self) -> float:
        """Devuelve el valor actual de la distancia máxima por frame en metros."""
        return self.MAX_DIST_PER_FRAME_M

    @property
    def min_dt_s(self) -> float:
        """Devuelve el valor actual del tiempo mínimo entre frames en segundos."""
        return self.MIN_DT_S
    
    @property
    def connection_manager(self) -> connection.ConnectionManager:
        return self._connection_manager
    
    @connection_manager.setter
    def connection_manager(self, connection_manager: connection.ConnectionManager) -> None:
        self._connection_manager = connection_manager
    
    @property
    def session(self) -> Session:
        return self._session
    
    @session.setter
    def session(self, session: Session) -> None:
        self._session = session
    
    @property
    def depth(self) -> float:
        return self._depth
    
    @depth.setter
    def depth(self, value: float) -> None:
        self._depth = value
    

    # --- Setters individuales ---
    @timestamp.setter
    def timestamp(self, value: float) -> None:
        """Actualiza el timestamp."""
        self._timestamp = float(value)

    @fps.setter
    def fps(self, value: float) -> None:
        """Actualiza los fps."""
        self._fps = float(value)
    
    @property
    def frame_size(self) -> tuple:
        return self._frame_size

    @frame_size.setter
    def frame_size(self, value: tuple) -> None:
        self._frame_size = value
    
    @property
    def video_anotator(self) -> anotator.VideoAnotator:
        return self._video_anotator
    
    @video_anotator.setter
    def video_anotator(self, value: anotator.VideoAnotator) -> None:
        self._video_anotator = value
    
    @property
    def detected_object(self) -> List[detected_object_data.DetectedObjectData]:
        return self._detected_object
    
    def add_detected_object(self, value: detected_object_data.DetectedObjectData) -> None:
        self._detected_object.append(value)
        
    def reset_detected_object(self) -> None:
        self._detected_object.clear()

    # --- Actualización simultánea ---
    def update(
        self, timestamp: Optional[float] = None, fps: Optional[float] = None
    ) -> None:
        """
        Permite actualizar timestamp y/o fps en una sola llamada.
        Si alguno de los argumentos es None, se mantiene el valor actual.
        """
        if timestamp is not None:
            self._timestamp = float(timestamp)
        if fps is not None:
            self._fps = float(fps)
            self.MAX_DIST_PER_FRAME_M = self.MAX_HUMAN_SPEED_KMH / 3.6 / self._fps
            self.MIN_DT_S = 1.0 / (2 * self._fps)
            
    def reset(self) -> None:
        self._timestamp = 0.0
        self._fps = 0.0
        self.MAX_DIST_PER_FRAME_M = self.MAX_HUMAN_SPEED_KMH / 3.6 / self._fps
        self.MIN_DT_S = 1.0 / (2 * self._fps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(timestamp={self._timestamp}, fps={self._fps})"
        )


globals = GlobalValuesStore()
