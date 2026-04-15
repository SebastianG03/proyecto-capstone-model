from typing import override

from sqlalchemy.exc import InvalidRequestError
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models.BallState import BallEventModel
from app.entities.models.HeatmapPoint import HeatmapPointModel
from app.logger import error_logger
import app.entities.utils.global_values_store as value_store

class TrackCollectionBall(RecordCollectionBase):
    orm_model = BallEventModel

    @override
    def generate_id(self, obj):
        return obj.frame_index

    @override
    def get_last(self) -> BallEventModel | None:
        db = value_store.globals.session
        try:
            return db.query(BallEventModel).order_by(BallEventModel.id.desc()).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[DBError] Error de consulta al obtener último registro de BallEventModel: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            return db.query(BallEventModel).order_by(BallEventModel.id.desc()).first()
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
        db = value_store.globals.session
        try:
            item = (
                db
                .query(BallEventModel)
                .filter(BallEventModel.track_id == track_id)
                .filter(BallEventModel.frame_index == frame_index)
            ).first()
            return item
        except InvalidRequestError as ie:
            error_logger.error(f"[DBError] Error de consulta al obtener registro para frame: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            return (
                db
                .query(BallEventModel)
                .filter(BallEventModel.track_id == track_id)
                .filter(BallEventModel.frame_index == frame_index)
            ).first()
        except Exception as e:
            error_logger.error(f"[DBError] Error al obtener registro para frame: {e}")
    
    def get_balls_last_frames(self, frame_num: int) -> list[BallEventModel]:
        db = value_store.globals.session
        
        try:
            return ( 
                    db.query(BallEventModel)
                    .filter(BallEventModel.frame_index <= frame_num)
                    .filter(BallEventModel.frame_index >= frame_num - 15)
                    .order_by(BallEventModel.frame_index.desc())
            ).all()
        except InvalidRequestError as ie:
            error_logger.error(f"[DBError] Error de consulta al obtener registros de BallEventModel: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            return (
                db.query(BallEventModel)
                .filter(BallEventModel.frame_index <= frame_num)
                .filter(BallEventModel.frame_index >= frame_num - 15)
                .order_by(BallEventModel.frame_index.desc())
            ).all()
        except Exception as e:
            error_logger.error(f"[DBError] Error al obtener registros de BallEventModel: {e}")
            return []

    @override
    def get_all(self) -> list[BallEventModel]:
        db = value_store.globals.session
        try:
            return (
                db
                .query(BallEventModel)
                .order_by(BallEventModel.frame_index.asc())
                .all()
            )
        except InvalidRequestError as ie:
            error_logger.error(f"[DBError] Error de consulta al obtener todos los registros de BallEventModel: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            return (
                db
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
        db = value_store.globals.session
        return (
            db
            .query(HeatmapPointModel)
            .filter(
                HeatmapPointModel.player_id == track_id,
                HeatmapPointModel.frame_number == frame_index,
            )
            .first()
        )
    
    @override
    def post(self, obj_data: dict):
        db = value_store.globals.session
        try:
            print(f"Creando nuevo registro con datos: {obj_data}")
            obj = HeatmapPointModel(**obj_data)
            db.add(obj)
            print(f"Objeto añadido a la sesión de la DB: {obj}")
            db.commit()
            db.refresh(obj)
            print(f"Objeto refrescado: {obj}")
            return obj
        except Exception as e:
            print(f"Error al crear registro: {e}")
            db.rollback()
            return None
    @override
    def get(self, obj_id: int) -> HeatmapPointModel | None:
        db = value_store.globals.session
        return db.query(HeatmapPointModel).filter(HeatmapPointModel.point_id == obj_id).first()
    
    def get_by_player_id(self, player_id: int) -> HeatmapPointModel:
        db = value_store.globals.session
        return db.query(HeatmapPointModel).filter(HeatmapPointModel.player_id == player_id).first()
    
    @override
    def get_all(self) -> list[HeatmapPointModel]:
        db = value_store.globals.session
        return db.query(HeatmapPointModel).all()

    @override
    def patch(self, obj_id: int, updates: dict):
        db = value_store.globals.session
        try:
            obj = db.query(HeatmapPointModel).filter(HeatmapPointModel.id == obj_id).first()
            
            if not obj:
                error_logger.error(f"Registro con ID {obj_id} no encontrado.")
                return None
            
            for key, val in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, val)
            
            db.flush()
            db.commit()
            db.refresh(obj)
            return obj
        except Exception as e:
            error_logger.error(f"Error al actualizar registro: {e}")
            db.rollback()
            return None

    
