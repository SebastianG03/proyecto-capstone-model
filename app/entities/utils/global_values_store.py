from typing import Optional

from app.entities.utils.singleton import Singleton

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

    # --- Getters ---
    @property
    def timestamp(self) -> float:
        """Devuelve el valor actual del timestamp."""
        return self._timestamp

    @property
    def fps(self) -> float:
        """Devuelve el valor actual de los fps."""
        return self._fps

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
    def update(self, timestamp: Optional[float] = None, fps: Optional[float] = None) -> None:
        """
        Permite actualizar timestamp y/o fps en una sola llamada.
        Si alguno de los argumentos es None, se mantiene el valor actual.
        """
        if timestamp is not None:
            self._timestamp = float(timestamp)
        if fps is not None:
            self._fps = float(fps)

    # --- Representación legible ---
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(timestamp={self._timestamp}, fps={self._fps})"
