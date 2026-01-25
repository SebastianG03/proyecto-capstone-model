from typing import override
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models.BallState import BallEventModel
from app.entities.models.HeatmapPoint import HeatmapPointModel


class TrackCollectionBall(RecordCollectionBase):
    orm_model = BallEventModel

    @override
    def generate_id(self, obj):
        return obj.frame_index

    @override
    def get_last(self) -> BallEventModel | None:
        return self.db.query(BallEventModel).order_by(BallEventModel.id.desc()).first()

    @override
    def get_record_for_frame(
        self, track_id: int, frame_index: int
    ) -> BallEventModel | None:
        """
        Busca un registro por track_id y frame_index.
        Puede ser sobrescrito si la colección usa otros campos.
        """
        try:
            item = (
                self.db
                .query(BallEventModel)
                .filter(BallEventModel.track_id == track_id)
                .filter(BallEventModel.frame_index == frame_index)
            ).first()
            return item
        except Exception as e:
            print(f"Error al obtener registro para frame: {e}")

    @override
    def get_all(self) -> list[BallEventModel]:
        return (
            self.db
            .query(BallEventModel)
            .order_by(BallEventModel.frame_index.asc())
            .all()
        )


class TrackCollectionHeatmapPoint(RecordCollectionBase):
    orm_model = HeatmapPointModel

    @override
    def generate_id(self, obj):
        return obj.point_id

    @override
    def get_record_for_frame(
        self, track_id: int, frame_index: int
    ) -> HeatmapPointModel | None:
        return (
            self.db
            .query(HeatmapPointModel)
            .filter(
                HeatmapPointModel.player_id == track_id,
                HeatmapPointModel.frame_number == frame_index,
            )
            .first()
        )
