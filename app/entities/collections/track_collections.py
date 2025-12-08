from typing import List, override
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import PlayerStateModel, HeatmapPointModel, BallEventModel

class TrackCollectionPlayer(RecordCollectionBase):
    orm_model = PlayerStateModel

    @override
    def generate_id(self, obj):
        return obj.track_id

    @override
    def get_last(self, db) -> PlayerStateModel:
        return db.query(PlayerStateModel).order_by(PlayerStateModel.id.desc()).first()

    @override
    def get_record_for_frame(self, track_id: int, frame_index: int):
        """
        Busca un registro por track_id y frame_index.
        Puede ser sobrescrito si la colección usa otros campos.
        """
        try:
            item = (self.db.query(PlayerStateModel)
                     .filter(PlayerStateModel.track_id == track_id)
                     .filter(PlayerStateModel.frame_index == frame_index)).first()
            return item
        except Exception as e:
            print(f"Error al obtener registro para frame: {e}")

    @override
    def get_all(self):
        return self.db.query(PlayerStateModel).order_by(PlayerStateModel.frame_index.asc()).all()


class TrackCollectionBall(RecordCollectionBase):
    orm_model = BallEventModel

    @override
    def generate_id(self, obj):
        return obj.frame_index
    @override
    def get_last(self, db):
        return db.query(BallEventModel).order_by(BallEventModel.id.desc()).first()

    @override
    def get_record_for_frame(self, track_id: int, frame_index: int):
        """
        Busca un registro por track_id y frame_index.
        Puede ser sobrescrito si la colección usa otros campos.
        """
        try:
            item = (self.db.query(BallEventModel)
                     .filter(BallEventModel.track_id == track_id)
                     .filter(BallEventModel.frame_index == frame_index)).first()
            return item
        except Exception as e:
            print(f"Error al obtener registro para frame: {e}")

    @override
    def get_all(self):
        return self.db.query(BallEventModel).order_by(BallEventModel.frame_index.asc()).all()

class TrackCollectionHeatmapPoint(RecordCollectionBase):
    orm_model = HeatmapPointModel

    @override
    def generate_id(self, obj):
        return obj.point_id
    
    @override
    def get_record_for_frame(self, track_id: int, frame_index: int):
        return (
            self.db.query(HeatmapPointModel)
            .filter(
                HeatmapPointModel.player_id == track_id,
                HeatmapPointModel.frame_number == frame_index
            )
            .first()
        )
