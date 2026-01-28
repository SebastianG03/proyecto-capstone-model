from typing import List, override

from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import Player, PlayerState


class TrackCollectionPlayer(RecordCollectionBase):
    orm_model = PlayerState

    @override
    def generate_id(self, obj):
        return obj.track_id

    @override
    def get_last(self) -> PlayerState | None:
        return self.db.query(PlayerState).order_by(PlayerState.id.desc()).first()

    def get_last_player(self, player_id: int) -> PlayerState | None:
        return (
            self.db
            .query(PlayerState)
            .filter(PlayerState.player_id == player_id)
            .order_by(PlayerState.timestamp_ms.desc())
            .first()
        )

    def get_player(self, player_id: int) -> Player | None:
        return (self.db.query(Player).filter(Player.player_id == player_id)).first()

    def get_player_id(self, player_id: int) -> int | None:
        query = (self.db.query(Player.id).filter(Player.player_id == player_id)).first()
        if not query:
            return -1
        value = query.tuple()
        return value[0]

    def get_player_states(self, player_id: int) -> List[PlayerState]:
        return (
            self.db
            .query(PlayerState)
            .filter(PlayerState.player_id == player_id)
            .order_by(PlayerState.frame_index.asc())
            .all()
        )

    def verify_player_exists(self, player_id: int) -> bool:
        player = self.get_player(player_id)
        return player is not None

    def verify_player_state_exists(self, player_id: int, frame_index: int) -> bool:
        state = self.get_record_for_frame(player_id, frame_index)
        return state is not None

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
        except Exception as e:
            print(f"Error al obtener registro para frame: {e}")

    @override
    def get_all(self):
        return self.db.query(Player).order_by(Player.id.asc()).all()

    def get_all_states(self) -> List[PlayerState]:
        return self.db.query(PlayerState).order_by(PlayerState.frame_index.asc()).all()

    def patch_state(self, player_id: int, frame_index: int, updates: dict):
        try:
            print(f"Actualizando registro ID {player_id} con {updates}")
            obj = (
                self.db
                .query(PlayerState)
                .filter(PlayerState.player_id == player_id)
                .filter(PlayerState.frame_index == frame_index)
                .first()
            )
            if not obj:
                print(
                    f"Registro con player_id {player_id} y frame_index {frame_index} no encontrado."
                )
                return None

            print(f"Objeto encontrado: {obj}")
            for key, value in updates.items():
                setattr(obj, key, value)
            # Flush antes de commit para detectar errores
            self.db.flush()
            self.db.commit()
            self.db.refresh(obj)
            print(f"Objeto actualizado: {obj}")
            return obj
        except Exception as e:
            print(f"Error al actualizar registro: {e}")
            self.db.rollback()
            return None
        finally:
            print(
                "Elementos actuales en base de datos PlayerState: ",
                len(self.get_all_states()),
            )

    def post_state(self, obj_data: dict):
        try:
            print(f"Creando nuevo registro de estado con datos: {obj_data}")
            obj = PlayerState(**obj_data)
            self.db.add(obj)
            print(f"Objeto añadido a la sesión de la DB: {obj}")
            # Flush antes de commit para detectar errores
            self.db.flush()
            self.db.commit()
            self.db.refresh(obj)
            print(f"Objeto refrescado: {obj}")
            return obj
        except Exception as e:
            print(f"Error al crear registro de estado: {e}")
            self.db.rollback()
            return None
        finally:
            print(
                "Elementos actuales en base de datos PlayerState: ",
                len(self.get_all_states()),
            )

    @override
    def post(self, obj_data: dict):
        try:
            print(
                f"[TrackCollectionPlayer] Creando nuevo registro con datos: {obj_data}"
            )
            obj = Player(**obj_data)
            self.db.add(obj)
            print(f"[TrackCollectionPlayer] Objeto añadido a la sesión de la DB: {obj}")
            # Flush antes de commit para detectar errores de integridad
            self.db.flush()
            self.db.commit()
            self.db.refresh(obj)
            print(f"[TrackCollectionPlayer] Objeto refrescado: {obj}")
            return obj
        except Exception as e:
            print(f"[TrackCollectionPlayer] Error al crear registro: {e}")
            self.db.rollback()
            return None
        finally:
            print(
                "[TrackCollectionPlayer] Elementos actuales en base de datos Jugador: ",
                len(self.get_all()),
            )

    @override
    def patch(self, obj_id: int, updates: dict):
        try:
            print(
                f"[TrackCollectionPlayer] Actualizando registro ID {obj_id} con {updates}"
            )
            obj = self.db.query(Player).filter(Player.id == obj_id).first()
            if not obj:
                print(
                    f"[TrackCollectionPlayer] Registro con ID {obj_id} no encontrado."
                )
                return None

            print(f"[TrackCollectionPlayer] Objeto encontrado: {obj}")
            for key, val in updates.items():
                print(
                    f"[TrackCollectionPlayer] Actualizando campo {key} con valor {val}"
                )
                if hasattr(obj, key):
                    setattr(obj, key, val)

            # Flush antes de commit para detectar errores de transacción
            self.db.flush()
            self.db.commit()
            print(f"[TrackCollectionPlayer] Objeto actualizado: {obj}")
            self.db.refresh(obj)
            print(f"[TrackCollectionPlayer] Objeto refrescado: {obj}")
            return obj
        except Exception as e:
            print(f"[TrackCollectionPlayer] Error al actualizar registro: {e}")
            self.db.rollback()
            return None
