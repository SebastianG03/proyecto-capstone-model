# record_collection_base.py
from sqlalchemy.orm import Session
from typing import Type, List, Optional

from app.entities.utils.singleton import Singleton


class RecordCollectionBase(metaclass=Singleton):
    """
    Clase base para manejar colecciones de registros en la DB.
    Puede ser heredada para diferentes modelos ORM.
    """
    orm_model: Type

    def __init__(self, db: Session):
        self.db = db
        if self.orm_model is None:
            raise ValueError("Debes definir 'orm_model' en la clase hija.")

    def generate_id(self, obj) -> int:
        """
        Genera el ID único usado en la base (depende del tipo de entidad).
        Se puede sobrescribir en la clase hija.
        """
        return obj.track_id
    
    def get_last(self, db):
        return db.query(self.orm_model).order_by(self.orm_model.id.desc()).first()

    #TODO Update to specific collection use
    def get_record_for_frame(self, track_id: int, frame_index: int):
        """
        Busca un registro por track_id y frame_index.
        Puede ser sobrescrito si la colección usa otros campos.
        """
        try:
            query = self.db.query(self.orm_model)

            if hasattr(self.orm_model, "track_id"):
                query = query.filter(self.orm_model.track_id == track_id)

            if hasattr(self.orm_model, "frame_index"):
                query = query.filter(self.orm_model.frame_index == frame_index)

            return query.first()
        except Exception as e:
            print(f"Error al obtener registro para frame: {e}")

    def get_by_id(self, obj_id: int):
        return self.db.query(self.orm_model).filter(self.orm_model.id == obj_id).first()

    def get(self, obj_id: int):
        return self.get_by_id(obj_id)

    def get_all(self) -> List:
        return self.db.query(self.orm_model).all()

    def post(self, obj_data: dict):
        try:
            print(f"Creando nuevo registro con datos: {obj_data}")
            obj = self.orm_model(**obj_data)
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
        finally:
            print(f'Elementos actuales en base de datos {self.orm_model.__name__}: ', len(self.get_all()))

    def patch(self, obj_id: int, updates: dict):
        try:
            print(f"Actualizando registro ID {obj_id} con {updates}")
            obj = self.get(obj_id)
            if not obj:
                print(f"Registro con ID {obj_id} no encontrado.")
                return None

            print(f"Objeto encontrado: {obj}")
            for key, val in updates.items():
                print(f"Actualizando campo {key} con valor {val}")
                if hasattr(obj, key):
                    setattr(obj, key, val)

            self.db.commit()
            print(f"Objeto actualizado: {obj}")
            self.db.refresh(obj)
            print(f"Objeto refrescado: {obj}")
            return obj
        except Exception as e:
            print(f"Error al actualizar registro: {e}")
            self.db.rollback()
            return None

    def delete(self, obj_id: int) -> bool:
        obj = self.get(obj_id)
        if not obj:
            return False

        self.db.delete(obj)
        self.db.commit()
        return True
