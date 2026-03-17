from typing import List, override

import numpy as np
from sqlalchemy.exc import InvalidRequestError

from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import Player, PlayerState
from app.entities.utils.global_values_store import globals
from app.logger import error_logger, info_logger

class TrackCollectionPlayer(RecordCollectionBase):
    orm_model = PlayerState

    @override
    def generate_id(self, obj):
        return obj.track_id

    @override
    def get_last(self) -> PlayerState | None:
        try:
            return self.db.query(PlayerState).order_by(PlayerState.id.desc()).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState: {ie}")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            last = self.db.query(PlayerState).order_by(PlayerState.id.desc()).first()
            return last
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener último registro de PlayerState: {e}")
            raise e

    def get_last_player(self, player_id: int) -> PlayerState | None:
        try:
            return (
                self.db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.timestamp_ms.desc())
                .first()
            )
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState: {ie}")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            last = (
                self.db
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
        try:
            return (self.db.query(Player).filter(Player.player_id == player_id)).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de Player: {ie}")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            player = (self.db.query(Player).filter(Player.player_id == player_id)).first()
            return player
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener jugador con player_id {player_id}: {e}")
            raise e


    def get_player_id(self, player_id: int) -> int | None:
        try:
            query = (self.db.query(Player.id).filter(Player.player_id == player_id)).first()
            if not query:
                return -1
            value = query.tuple()
            return value[0]
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de Player para obtener ID: {ie}")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            query = (self.db.query(Player.id).filter(Player.player_id == player_id)).first()
            if not query:
                return -1
            value = query.tuple()
            return value[0]
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener ID de jugador con player_id {player_id}: {e}")
            raise e

    def get_player_states(self, player_id: int) -> List[PlayerState]:
        try:
            return (
                self.db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState para obtener estados: {ie}")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            states = (
                self.db
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
        try:
            player_states = (
                self.db
                .query(PlayerState.distance)
                .filter(PlayerState.player_id == player_id)
                .order_by(PlayerState.frame_index.desc())
                .all()
            )
            
            np_states =np.array(player_states)
            return np_states.sum()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error al refrescar la sesión de PlayerState para calcular distancia total: {ie}")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            player_states = (
                self.db
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
        try:
            return (
                self.db
                .query(PlayerState)
                .filter(PlayerState.player_id == track_id)
                .filter(PlayerState.frame_index == frame_index)
            ).first()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener registro para frame: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            record = self.get_record_for_frame(track_id, frame_index)
            return record
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener registro para frame: {e}")
            raise e

    @override
    def get_all(self):
        return self.db.query(Player).order_by(Player.id.asc()).all()

    def get_all_states(self) -> List[PlayerState]:
        try:
            return self.db.query(PlayerState).order_by(PlayerState.frame_index.asc()).all()
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al obtener todos los estados de jugadores: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            states = self.db.query(PlayerState).order_by(PlayerState.frame_index.asc()).all()
            return states
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al obtener todos los estados de jugadores: {e}")
            raise e
        
    def _patch_state(self, player_id: int, frame_index: int, updates: dict):
        print(f"Actualizando registro ID {player_id} con {updates}")
        obj = (
            self.db
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
        self.db.flush()
        self.db.commit()
        self.db.refresh(obj)
        info_logger.info(f"[PlayerDB] Objeto actualizado: {obj}")
        return obj

    def patch_state(self, player_id: int, frame_index: int, updates: dict):
        try:
            obj = self._patch_state(player_id, frame_index, updates)
            return obj
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al actualizar registro: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            state = self._patch_state(player_id, frame_index, updates)
            return state
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al actualizar registro: {e}")
            self.db.rollback()
            return None
        finally:
            error_logger.info(
                "Elementos actuales en base de datos PlayerState: ",
                len(self.get_all_states()),
            )
            
    def _post_state(self, obj_data: dict):
        obj = PlayerState(**obj_data)
        self.db.add(obj)
        self.db.flush()
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def post_state(self, obj_data: dict):
        try:
            return self._post_state(obj_data)
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al crear registro de estado: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            state = self._post_state(obj_data)
            return state
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al crear registro de estado: {e}")
            self.db.rollback()
            return None
        finally:
            info_logger.info(
                f"[PlayerDB] Elementos actuales en base de datos PlayerState: {len(self.get_all_states())}"
            )

    def _post(self, obj_data: dict):
        obj = Player(**obj_data)
        self.db.add(obj)
        self.db.flush()
        self.db.commit()
        info_logger.info(f"[PlayerDB] Objeto añadido a la sesión de la DB: {obj}")
        self.db.refresh(obj)
        return obj

    @override
    def post(self, obj_data: dict):
        try:
            return self._post(obj_data)
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al crear registro de jugador: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            player = self._post(obj_data)
            return player
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al crear registro de jugador: {e}")
            self.db.rollback()
            return None
        finally:
            info_logger.info(
                f"[PlayerDB] Elementos actuales en base de datos Jugador: {len(self.get_all())}"
            )

    def _path(self, obj_id: int, updates: dict):
        obj = self.db.query(Player).filter(Player.id == obj_id).first()
        if not obj:
            info_logger.info(f"[PlayerDB] Registro con ID {obj_id} no encontrado.")
            return None

        for key, value in updates.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        self.db.flush()
        self.db.commit()
        self.db.refresh(obj)
        return obj

    @override
    def patch(self, obj_id: int, updates: dict):
        try:
            return self._path(obj_id, updates)
        except InvalidRequestError as ie:
            error_logger.error(f"[PlayerDBError] Error de consulta al actualizar registro de jugador: {ie}, iniciando refresco de la sesión y reintentando.")
            self.db.rollback()
            self.db = globals.connection_manager.create_session()
            player = self._path(obj_id, updates)
            return player
        except Exception as e:
            error_logger.error(f"[PlayerDBError] Error al actualizar registro: {e}")
            self.db.rollback()
            return None
