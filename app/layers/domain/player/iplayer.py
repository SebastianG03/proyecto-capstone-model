from abc import ABC
from typing import List, Optional, Tuple


class IPlayer(ABC):
    id: int
    bbox: List[float]
    position_2d: Tuple[float, float]
    position_adjusted_2d: Tuple[float, float]
    position_transformed_2d: Optional[List[float]]
    position_3d: Optional[List[float]]
    position_adjusted_3d: Optional[List[float]]
    position_transformed_3d: Optional[List[float]]