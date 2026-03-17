from typing import Optional

from app.entities.utils.singleton import Singleton
from app.infraestructure.database.connection_manager import ConnectionManager


class GlobalValuesStore(metaclass=Singleton):
    """
    Almacena y permite actualizar dos valores: timestamp (float) y fps (float).
    """

    def __init__(self, timestamp: float = 0.0, fps: float = 0.0) -> None:
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
        self._connection_manager: ConnectionManager

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
    def connection_manager(self) -> ConnectionManager:
        return self._connection_manager
    
    @connection_manager.setter
    def connection_manager(self, connection_manager: ConnectionManager) -> None:
        self._connection_manager = connection_manager
    

    # --- Setters individuales ---
    @timestamp.setter
    def timestamp(self, value: float) -> None:
        """Actualiza el timestamp."""
        self._timestamp = float(value)

    @fps.setter
    def fps(self, value: float) -> None:
        """Actualiza los fps."""
        self._fps = float(value)

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
            self.MAX_DIST_PER_FRAME_M = self.MAX_HUMAN_SPEED_KMH / 3.6 / 24
            self.MIN_DT_S = 1.0 / (2 * 24)
            
    def reset(self) -> None:
        self._timestamp = 0.0
        self._fps = 0.0
        self.MAX_DIST_PER_FRAME_M = self.MAX_HUMAN_SPEED_KMH / 3.6 / 24
        self.MIN_DT_S = 1.0 / (2 * 24)
        

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(timestamp={self._timestamp}, fps={self._fps})"
        )


globals = GlobalValuesStore()
