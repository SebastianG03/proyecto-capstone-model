# tracker_service.py
import logging
from typing import List, Union

import supervision as sv
from cv2.typing import MatLike
from sqlalchemy.orm import Session
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.interfaces.tracker_service_base import TrackerServiceBase

class TrackerService(TrackerServiceBase):
    """
    Implementación concreta del servicio de tracking, preparada para streaming.
    - Mantiene self.last_tracked para acceso externo (p.ej. TeamAssigner)
    """

    def __init__(self, model_path: str, use_half_precision: bool = False):
        super().__init__(model_path=model_path, use_half_precision=use_half_precision)
        self.last_tracked: sv.Detections | None = None


    def get_object_tracks(
        self,
        frames: Union[List[MatLike], MatLike],
        frame_num: int,
        db: Session
    ):
        """
        Compatibilidad con API anterior que pasaba una lista de frames.
        En streaming, se le puede pasar un único frame.
        """
        print("TrackerService.get_object_tracks called.")
        if isinstance(frames, list):
            print(f"Procesando lista de {len(frames)} frames...")
            for i, frame in enumerate(frames):
                self.process_frame(frame, i, db)
        else:
            print("Procesando un solo frame...")
            # Un solo frame — mantenemos frame_num = 0 si no se especifica
            self.process_frame(frame=frames, db=db, frame_num=frame_num)


    def get_tracker(self, key: str):
        from app.modules.trackers.tracker_factory import TrackerFactoryError
        try:
            tracker = self.tracker_factory.get_tracker(key)
            if not tracker:
                raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
            return tracker
        except Exception as e:
            logging.exception(f"Error getting tracker {key}: {e}")
            print(f"Error getting tracker {key}: {e}")
            raise e

    