from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from numpy.typing import NDArray
from sqlalchemy import Boolean, Column, Integer, Float, String, ForeignKey
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.infraestructure.database.connection_manager import Base
import json


class Player(Base):
    """
    Informacion INMUTABLE (o que cambia muy poco) del jugador.
    """

    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)

    # --- datos de identidad --------------------------------------------------
    player_id = Column(Integer, unique=True, nullable=False, index=True)
    team = Column(String, nullable=True, default=None)
    color = Column(String, nullable=True, default=None)
    shirt_number = Column(Integer, nullable=True, default=None)
    goals = Column(Integer, default=0)

    # --- timestamps de auditoria --------------------------------------------
    created_at = Column(DATETIME(timezone=True), server_default=func.now())
    updated_at = Column(
        DATETIME(timezone=True),
        nullable=False,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )

    # relacion 1-N con sus estados
    states = relationship("PlayerState", back_populates="player")

    def to_dict(self):
        return {
            "id": self.id,
            "player_id": self.player_id,
            "team": self.team,
            "color": self.color,
            "shirt_number": self.shirt_number,
            "goals": self.goals,
            "created_at": self.created_at,
            "updated_ut": self.updated_at,
        }


class PlayerState(Base):
    """
    Estado del jugador FRAME-A-FRAME.
    """

    __tablename__ = "player_states"

    id = Column(Integer, primary_key=True, index=True)

    player_id = Column(
        Integer,
        ForeignKey("players.player_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    frame_index = Column(Integer, index=True, nullable=False)
    bbox = Column(String, nullable=True)  # JSON list
    conf = Column(Float)

    # posicion
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

    x_transformed = Column(Float, nullable=True)
    y_transformed = Column(Float, nullable=True)

    x_smoothed = Column(Float, nullable=True)
    y_smoothed = Column(Float, nullable=True)

    # balon
    ball_x = Column(Float, nullable=True)
    ball_y = Column(Float, nullable=True)
    ball_z = Column(Float, nullable=True)

    has_ball = Column(Boolean, default=False)
    ball_possession_time = Column(Float, default=0.0)
    ball_owner_id = Column(Integer, index=True, nullable=True)

    # dinamica
    distance = Column(Float, default=0.0)  # on meters
    incremental_distance = Column(Float, default=0.0)
    speed = Column(Float, default=0.0)  # on km per hour
    acceleration = Column(Float, default=0.0)
    is_sprint = Column(Boolean, default=False)

    time_visible = Column(Float, default=0.0)

    # timestamp absoluto guardado en milisegundos
    timestamp_ms = Column(Float, index=True, nullable=True)

    # timestamp de insercion
    created_at = Column(DATETIME(timezone=True), default=datetime.now(timezone.utc))

    # relacion inversa
    player = relationship("Player", back_populates="states")

    def set_bbox(self, bbox_list: list[int]):
        self.bbox = json.dumps(bbox_list)

    def get_bbox(self):
        if self.bbox is None:
            return None
        try:
            return json.loads(f"{self.bbox}")
        except Exception:
            return None

    def to_dict(self):
        return {
            "id": self.id,
            "player_id": self.player_id,
            "frame_index": self.frame_index,
            "bbox": self.get_bbox(),
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "x_transformed": self.x_transformed,
            "y_transformed": self.y_transformed,
            "x_smoothed": self.x_smoothed,
            "y_smoothed": self.y_smoothed,
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
            "created_at": self.created_at.isoformat(),
        }



@dataclass
class State:
    def __init__(
        self,
        bbox: list[float],
        x: float,
        y: float,
        conf: float,
        timestamp: float,
        player_id: int,
        frame_num: int):
        self.bbox = bbox
        self.x = x
        self.y = y
        self.conf = conf
        self.timestamp = timestamp
        self.player_id = player_id
        self.frame_num = frame_num
    
    @staticmethod
    def to_instance(payload: dict[str, Any]):
        return State(
            bbox=payload["bbox"],
            x=payload["x"],
            y=payload["y"],
            conf=payload["conf"],
            timestamp=payload["timestamp_ms"],
            player_id=payload["player_id"],
            frame_num=payload["frame_index"],
        )

@dataclass
class PlayerStatus:
    player_id: int
    frame_index: int
    vx: float
    vy: float
    speed: float
    direction: float
    time: float
    delta_x: NDArray
    xo: NDArray
    xf: NDArray
