from typing import List, override

import numpy as np
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import Query, Session

from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import Player, PlayerState
import app.entities.utils.global_values_store as value_store
from app.logger import error_logger, info_logger

class TrackCollectionPlayer(RecordCollectionBase):
    orm_model = PlayerState

    @override
    def generate_id(self, obj):
        return obj.track_id

    @override
    def get_last(self) -> PlayerState | None:
        db = value_store.globals.session
        try:
            return db.query(PlayerState).order_by(PlayerState.id.desc()).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState: {ie}")
            db = value_store.globals.connection_manager.create_session()
            last = db.query(PlayerState).order_by(PlayerState.id.desc()).first()
            return last
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener último registro de PlayerState: {e}")
            raise e

    def get_last_player(self, player_id: int) -> PlayerState | None:
        db = value_store.globals.session
        try:
            return (
                db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.timestamp_ms.desc())
                .first()
            )
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState: {ie}")
            db = value_store.globals.connection_manager.create_session()
            last = (
                db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.timestamp_ms.desc())
                .first()
            )
            return last
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener último registro de PlayerState para player_id {player_id}: {e}")
            raise e

    def get_player(self, player_id: int) -> Player | None:
        db = value_store.globals.session
        try:
            return (db.query(Player).filter(Player.player_id == player_id)).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de Player: {ie}")
            db.rollback()
            db = value_store.globals.connection_manager.create_session()
            player = (db.query(Player).filter(Player.player_id == player_id)).first()
            return player
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener jugador con player_id {player_id}: {e}")
            raise e


    def get_player_id(self, player_id: int) -> int | None:
        db = value_store.globals.session
        try:
            query = (db.query(Player.id).filter(Player.player_id == player_id)).first()
            if not query:
                return -1
            value = query.tuple()
            return value[0]
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de Player para obtener ID: {ie}")
            db = value_store.globals.connection_manager.create_session()
            query = (db.query(Player.id).filter(Player.player_id == player_id)).first()
            if not query:
                return -1
            value = query.tuple()
            return value[0]
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener ID de jugador con player_id {player_id}: {e}")
            raise e

    def get_player_states(self, player_id: int) -> List[PlayerState]:
        db = value_store.globals.session
        try:
            return (
                db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState para obtener estados: {ie}")
            db = value_store.globals.connection_manager.create_session()
            states = (
                db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
            return states
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener estados de jugador con player_id {player_id}: {e}")
            raise e
        
    def calculate_player_total_distance(self, player_id: int) -> float:
        db = value_store.globals.session
        try:
            player_states = (
                db
                .query(PlayerState.distance)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
            
            np_states = np.array(player_states)
            return np_states.sum()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState para calcular distancia total: {ie}")
            db = value_store.globals.connection_manager.create_session()
            player_states = (
                db
                .query(PlayerState.distance)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
            np_states =np.array(player_states)
            return np_states.sum()
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al calcular distancia total de jugador con player_id {player_id}: {e}")
            raise e

    def verify_player_exists(self, player_id: int) -> bool:
        try:
            player = self.get_player(player_id)
            return player is not None
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al verificar existencia de jugador con player_id {player_id}: {e}")
            raise e

    def verify_player_state_exists(self, player_id: int, frame_index: int) -> bool:
        try:
            state = self.get_record_for_frame(player_id, frame_index)
            return state is not None
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al verificar existencia de estado de jugador con player_id {player_id} y frame_index {frame_index}: {e}")
            raise e

    @override
    def get_record_for_frame(
        self, track_id: int, frame_index: int
    ) -> PlayerState | None:
        """
        Busca un registro por track_id y frame_index.
        Puede ser sobrescrito si la colección usa otros campos.
        """
        db = value_store.globals.session
        try:
            return (
                db
                .query(PlayerState)
                .filter(PlayerState.player_id == track_id)
                .filter(PlayerState.frame_index == frame_index)
            ).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener registro para frame: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            record = self.get_record_for_frame(track_id, frame_index)
            return record
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener registro para frame: {e}")
            raise e
    
    def get_state_by_id(self, state_id: int):
        db = value_store.globals.session
        try:
            return db.query(PlayerState).filter(PlayerState.id == state_id).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener registro por id: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            return db.query(PlayerState).filter(PlayerState.id == state_id).first()
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener registro por id: {e}")
            raise e

    def get_states_by_frame(self, frame_num: int):
        try:
            db = value_store.globals.session
            return db.query(PlayerState).filter(PlayerState.frame_index == frame_num).all()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener registro por frame: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            return db.query(PlayerState).filter(PlayerState.frame_index == frame_num).all()
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener registro por frame: {e}")
            raise e

    @override
    def get_all(self):
        db = value_store.globals.session
        return db.query(Player).order_by(Player.id.asc()).all()
    
    def _get_query_all_states(self, db: Session, frame_index: int = 0) -> Query[PlayerState]:
            query = db.query(PlayerState).order_by(PlayerState.frame_index.desc())
            if frame_index > 0:
                query = (
                    db.query(PlayerState)
                    .filter(PlayerState.frame_index >= frame_index - 24 * 5)
                    .order_by(PlayerState.frame_index.desc()))
            
            return query

    def get_all_states(self, frame_index: int = 0) -> List[PlayerState]:
        db = value_store.globals.session
        try:
            query = self._get_query_all_states(db, frame_index) 
            return query.all()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener todos los estados de jugadores: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            query = self._get_query_all_states(db, frame_index)
            return query.all()
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener todos los estados de jugadores: {e}")
            raise e
    def _get_previous_state(self, player_id: int, frame_index: int, db: Session) -> PlayerState | None:
            if frame_index < 24:
                return None
            return (
                db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .filter(PlayerState.frame_index <= frame_index - 25)
                .order_by(PlayerState.frame_index.desc())
            ).first()
    
    def get_states_previous_frame(self, frame_index: int) -> List[PlayerState]:
        previous_frame = frame_index - value_store.globals.fps
        try:
            db = value_store.globals.session
            query = (db.query(PlayerState)
                     .filter(PlayerState.frame_index < previous_frame)
                     .filter(PlayerState.frame_index >= previous_frame - 15)
                     .order_by(PlayerState.frame_index.desc())
                     ).all()
            return query
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener estados previos: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            query = (db.query(PlayerState)
                     .filter(PlayerState.frame_index < previous_frame)
                     .filter(PlayerState.frame_index >= previous_frame - 5) 
                     .group_by(PlayerState.player_id)
                     .order_by(PlayerState.frame_index.desc())
                     ).all()
            return query
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener estados previos: {e}")
            raise e

    def get_previous_state(self, player_id: int, frame_index: int) -> PlayerState | None:
        try:
            db = value_store.globals.session
            return self._get_previous_state(player_id, frame_index, db)
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener estado previo: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            state = self._get_previous_state(player_id, frame_index, db)
            return state
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener estado previo: {e}")
            raise e
    
    def _patch_state(self, player_id: int, frame_index: int, updates: dict):
        db = value_store.globals.session
        info_logger.info(f"Actualizando registro ID {player_id} con {updates}")
        obj = (
            db
            .query(PlayerState)
            .filter(PlayerState.player_id == player_id)
            .filter(PlayerState.frame_index == frame_index)
            .first()
        )
        if not obj:
            info_logger.info(
                f"[PlayerDB] Registro con player_id {player_id} y frame_index {frame_index} no encontrado."
            )
            return None

        for key, value in updates.items():
            setattr(obj, key, value)
        db.flush()
        db.commit()
        db.refresh(obj)
        info_logger.info(f"[PlayerDB] Objeto actualizado: {obj}")
        return obj

    def patch_state(self, player_id: int, frame_index: int, updates: dict):
        db = value_store.globals.session
        try:
            obj = self._patch_state(player_id, frame_index, updates)
            return obj
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al actualizar registro: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            value_store.globals.session = db
            state = self._patch_state(player_id, frame_index, updates)
            return state
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al actualizar registro: {e}")
            db.rollback()
            return None
        finally:
            error_logger.info(
                f"Elementos actuales en base de datos PlayerState: {len(self.get_all_states())}"
            )

    
    def _post_state(self, obj_data: dict):
        db = value_store.globals.session
        obj = PlayerState(**obj_data)
        db.add(obj)
        db.flush()
        db.commit()
        db.refresh(obj)
        return obj

    def post_state(self, obj_data: dict):
        db = value_store.globals.session
        try:
            return self._post_state(obj_data)
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al crear registro de estado: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            value_store.globals.session = db
            state = self._post_state(obj_data)
            return state
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al crear registro de estado: {e}")
            db.rollback()
            return None
        finally:
            info_logger.info(
                f"[PlayerDB] Elementos actuales en base de datos PlayerState: {len(self.get_all_states())}"
            )

    def _post(self, obj_data: dict):
        db = value_store.globals.session
        obj = Player(**obj_data)
        db.add(obj)
        db.flush()
        db.commit()
        info_logger.info(f"[PlayerDB] Objeto añadido a la sesión de la DB: {obj}")
        db.refresh(obj)
        return obj

    @override
    def post(self, obj_data: dict):
        db = value_store.globals.session
        try:
            return self._post(obj_data)
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al crear registro de jugador: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            value_store.globals.session = db
            player = self._post(obj_data)
            return player
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al crear registro de jugador: {e}")
            db.rollback()
            return None
        finally:
            info_logger.info(
                f"[PlayerDB] Elementos actuales en base de datos Jugador: {len(self.get_all())}"
            )

    def _patch(self, obj_id: int, updates: dict):
        db = value_store.globals.session
        obj = db.query(Player).filter(Player.player_id == obj_id).first()
        if not obj:
            info_logger.info(f"[PlayerDB] Registro con ID {obj_id} no encontrado.")
            return None

        for key, value in updates.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        db.flush()
        db.commit()
        db.refresh(obj)
        return obj

    @override
    def patch(self, obj_id: int, updates: dict):
        db = value_store.globals.session
        try:
            return self._patch(obj_id, updates)
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al actualizar registro de jugador: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            value_store.globals.session = db
            player = self._patch(obj_id, updates)
            return player
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al actualizar registro: {e}")
            db.rollback()
            return None

    def get_player_speeds(self, player_id: int, max_speed: float) -> List[float]:
        db = value_store.globals.session
        try:
            speeds = (
                db
                .query(PlayerState.speed)
                .filter(PlayerState.player_id == player_id)
                .filter(PlayerState.speed <= max_speed)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
            return [float(s[0]) for s in speeds if s[0] is not None]
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener velocidades de jugador: {ie}, iniciando refresco de la sesión y reintentando.")
            db = value_store.globals.connection_manager.create_session()
            speeds = (
                db
                .query(PlayerState.speed)
                .filter(PlayerState.player_id == player_id)
                .filter(PlayerState.speed <= max_speed)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
            return [float(s[0]) for s in speeds if s[0] is not None]
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener velocidades de jugador con player_id {player_id}: {e}")
            raise e
