import json
from sqlalchemy import Column, DateTime, Integer, Float, String, func
from app.modules.services.database import Base

class BallEventModel(Base):
    """
    Eventos o detecciones del balón almacenadas (opcional).
    """
    __tablename__ = "ball_event"
    id = Column(Integer, primary_key=True, index=True)
    frame_index = Column(Integer, index=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

    # Vectorial Velocity
    vx = Column(Float, default=0.0)
    vy = Column(Float, default=0.0)

    # transformed position
    x_transformed = Column(Float, nullable=True)
    y_transformed = Column(Float, nullable=True)
    
    bbox = Column(String, nullable=True)
    owner_id = Column(Integer, index=True, nullable=True)
    track_id = Column(Integer, index=True, default=0)
    
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
            "frame_index": self.frame_index,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "owner_id": self.owner_id,
            "track_id": self.track_id,
            "bbox": self.get_bbox(),
            "x_transformed": self.x_transformed,
            "y_transformed": self.y_transformed
        }
