from sqlalchemy import Column, DateTime, Integer, Float, func
from app.modules.services.database import Base

class HeatmapPointModel(Base):
    __tablename__ = "heatmap_point"
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, index=True)
    frame_number = Column(Integer, index=True)
    x = Column(Float)
    z = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
