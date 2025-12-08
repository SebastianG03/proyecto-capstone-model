from sqlalchemy import Boolean, Column, Integer, Float, String
from app.modules.services.database import Base
import json


class PlayerStateModel(Base):
    __tablename__ = "player_state"

    id = Column(Integer, primary_key=True, index=True)

    # Identidad del jugador
    player_id = Column(Integer, index=True, nullable=False)
    team = Column(String)
    color = Column(String)
    shirt_number = Column(Integer, nullable=True)

    # Frame
    frame_index = Column(Integer, index=True)

    # Bounding box crudo del detector (xmin, ymin, xmax, ymax)
    bbox = Column(String, nullable=True)  # JSON encoded list

    # Posición del jugador
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    
    # transformed position
    x_transformed = Column(Float, nullable=True)
    y_transformed = Column(Float, nullable=True)
    
    #Smoothed position
    x_smoothed = Column(Float, nullable=True)
    y_smoothed = Column(Float, nullable=True)

    # Balón
    ball_x = Column(Float, nullable=True)
    ball_y = Column(Float, nullable=True)
    ball_z = Column(Float, nullable=True)

    has_ball = Column(Boolean, default=False)
    ball_possession_time = Column(Float, default=0.0)

    # NEW → necesario para detectar candidatos a pase
    ball_owner_id = Column(Integer, index=True, nullable=True)

    # Dinámica del jugador
    distance = Column(Float, default=0.0)                 # Distancia total
    incremental_distance = Column(Float, default=0.0)     # Distancia del frame
    speed = Column(Float, default=0.0)
    acceleration = Column(Float, default=0.0)
    is_sprint = Column(Boolean, default=False)

    time_visible = Column(Float, default=0.0)

    # Timestamp relativo al video
    timestamp_ms = Column(Float, index=True, nullable=True)

    # -----------------------
    # HELPERS
    # -----------------------

    def set_bbox(self, bbox_list: list[int]):
        """Convierte lista Python → string JSON seguro"""
        self.bbox = json.dumps(bbox_list)

    def get_bbox(self):
        """Convierte JSON almacenado → lista Python"""
        if self.bbox is None:
            return None

        try:
            bbox = f'{self.bbox}'
            return json.loads(bbox)
        except Exception:
            return None

    def to_dict(self):
        return {
            "id": self.id,
            "player_id": self.player_id,
            "team": self.team,
            "color": self.color,
            "shirt_number": self.shirt_number,
            "frame_index": self.frame_index,
            "bbox": self.get_bbox(),
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_z": self.ball_z,
            "has_ball": self.has_ball,
            "ball_possession_time": self.ball_possession_time,
            "ball_owner_id": self.ball_owner_id,
            "distance": self.distance,
            "incremental_distance": self.incremental_distance,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "is_sprint": self.is_sprint,
            "time_visible": self.time_visible,
            "timestamp_ms": self.timestamp_ms,
        }
