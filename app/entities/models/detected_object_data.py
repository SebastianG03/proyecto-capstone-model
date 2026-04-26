from dataclasses import dataclass
from sqlalchemy import Column, Float, Integer, String
from app.infraestructure.database.connection_manager import Base

@dataclass
class DetectedObjectData:
    name: str
    id: int
    confidence: float


class AnalysisData(Base):
    __tablename__ = "analysis_data"
    id = Column(Integer, primary_key=True, index=True)
    frame = Column(Integer, index=True, unique=False)
    track_id = Column(Integer, index=True, unique=False)
    x = Column(Float(precision=6), default=None)
    y = Column(Float(precision=6), default=None)
    vclass = Column(String, default=None)
    shirt_number = Column(Integer, default=None)
    velocity = Column(Float(precision=6), default=None)
    timestamps = Column(Float(precision=8))
    
    def to_dict(self):
        return {
            "frame": self.frame,
            "track_id": self.track_id,
            "x": self.x,
            "y": self.y,
            "vclass": self.vclass,
            "shirt_number": self.shirt_number,
            "velocity": self.velocity,
            "timestamps": self.timestamps,
        }
