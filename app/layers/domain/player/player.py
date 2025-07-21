from ast import List
from dataclasses import field
import hashlib
from typing import Any, Dict, Optional, Tuple

from app.layers.domain.player.iplayer import IPlayer


class Player(IPlayer):
    speed: Optional[float]
    distance: Optional[float] 
    team: str
    team_color: Optional[List[float]] # RBG color
    has_ball: bool
    timestamp: float = field(default_factory=lambda: 0.0)
    
    def __post_init__(self):
        """Validación y procesamiento post-inicialización"""
        if self.speed is not None and self.speed < 0:
            raise ValueError("Speed cannot be negative")
        if self.distance is not None and self.distance < 0:
            raise ValueError("Distance cannot be negative")
    
    def get_hash(self) -> str:
        """Genera un hash único para el jugador basado en sus propiedades"""
        data = f"{self.id}_{self.position}_{self.team_id}_{self.timestamp}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_center_position(self) -> Tuple[float, float]:
        """Calcula la posición central del bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_bbox_area(self) -> float:
        """Calcula el área del bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el jugador a diccionario"""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'position': self.position,
            'position_adjusted': self.position_adjusted,
            'position_transformed': self.position_transformed,
            'speed': self.speed,
            'distance': self.distance,
            'team_id': self.team_id,
            'team_color': self.team_color,
            'timestamp': self.timestamp
        }