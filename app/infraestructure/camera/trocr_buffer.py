from __future__ import annotations
import numpy as np
from typing import List, Optional, Callable

class TROCRBuffer:
    """Buffer circular de crops 60×60.  El flush() se hace fuera."""
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self._clear()

    def _clear(self):
        self.crops: List[np.ndarray] = []
        self.cbks : List[Callable[[Optional[int], float], None]] = []

    def push(self,
             crop: np.ndarray,
             callback: Callable[[Optional[int], float], None]) -> bool:
        """True -> el llamador debe hacer flush()."""
        self.crops.append(crop)
        self.cbks.append(callback)
        return len(self.crops) >= self.batch_size

    def flush(self) -> tuple[list[np.ndarray], list[Callable]]:
        crops, cbks = self.crops.copy(), self.cbks.copy()
        self._clear()
        return crops, cbks