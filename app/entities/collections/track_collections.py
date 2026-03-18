from typing import override

from sqlalchemy.exc import InvalidRequestError
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models.BallState import BallEventModel
from app.entities.models.HeatmapPoint import HeatmapPointModel
from app.logger import error_logger
from app.entities.utils.global_values_store import globals

class TrackCollectionBall(RecordCollectionBase):
    orm_model = BallEventModel

    @override
    def generate_id(self, obj):
        return obj.frame_index

    @override
    def get_last(self) -> BallEventModel | None:
        try:
            return self.db.query(BallEventModel).order_by(BallEventModel.id.desc()).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[DBError] Error de consulta al obtener último registro de BallEventModel: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db = globals.connection_manager.create_session()
            return self.db.query(BallEventModel).order_by(BallEventModel.id.desc()).first()
        except Exception as e:
            error_logger.error(f"[DBError] Error al obtener último registro de BallEventModel: {e}")
            return None

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
        except InvalidRequestError as ie:
            error_logger.error(f"[DBError] Error de consulta al obtener registro para frame: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db = globals.connection_manager.create_session()
            return (
                self.db
                .query(BallEventModel)
                .filter(BallEventModel.track_id == track_id)
                .filter(BallEventModel.frame_index == frame_index)
            ).first()
        except Exception as e:
            error_logger.error(f"[DBError] Error al obtener registro para frame: {e}")

    @override
    def get_all(self) -> list[BallEventModel]:
        try:
            return (
                self.db
                .query(BallEventModel)
                .order_by(BallEventModel.frame_index.asc())
                .all()
            )
        except InvalidRequestError as ie:
            error_logger.error(f"[DBError] Error de consulta al obtener todos los registros de BallEventModel: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db = globals.connection_manager.create_session()
            return (
                self.db
                .query(BallEventModel)
                .order_by(BallEventModel.frame_index.asc())
                .all()
            )
        except Exception as e:
            error_logger.error(f"[DBError] Error al obtener todos los registros de BallEventModel: {e}")
            return []


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
    
    @override
    def post(self, obj_data: dict):
        try:
            print(f"Creando nuevo registro con datos: {obj_data}")
            obj = HeatmapPointModel(**obj_data)
            self.db.add(obj)
            print(f"Objeto añadido a la sesión de la DB: {obj}")
            self.db.commit()
            self.db.refresh(obj)
            print(f"Objeto refrescado: {obj}")
            return obj
        except Exception as e:
            print(f"Error al crear registro: {e}")
            self.db.rollback()
            return None
    @override
    def get(self, obj_id: int) -> HeatmapPointModel | None:
        return self.db.query(HeatmapPointModel).filter(HeatmapPointModel.point_id == obj_id).first()
    
    def get_by_player_id(self, player_id: int) -> HeatmapPointModel:
        return self.db.query(HeatmapPointModel).filter(HeatmapPointModel.player_id == player_id).first()
    
    @override
    def get_all(self) -> list[HeatmapPointModel]:
        return self.db.query(HeatmapPointModel).all()

    @override
    def patch(self, obj_id: int, updates: dict):
        try:
            obj = self.db.query(HeatmapPointModel).filter(HeatmapPointModel.id == obj_id).first()
            
            if not obj:
                error_logger.error(f"Registro con ID {obj_id} no encontrado.")
                return None
            
            for key, val in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, val)
            
            self.db.flush()
            self.db.commit()
            self.db.refresh(obj)
            return obj
        except Exception as e:
            error_logger.error(f"Error al actualizar registro: {e}")
            self.db.rollback()
            return None

    
