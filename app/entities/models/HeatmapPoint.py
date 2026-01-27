from sqlalchemy import Column, DateTime, Integer, Float, String, func
from app.infraestructure.services.database import Base


class HeatmapPointModel(Base):
    __tablename__ = "heatmap_point"
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, index=True)
    path = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
