# tracker_service.py
import logging
from typing import List, Union

import supervision as sv
from cv2.typing import MatLike
from sqlalchemy.orm import Session
from app.entities.interfaces.tracker_service_base import TrackerServiceBase


class TrackerService(TrackerServiceBase):
    """
    Implementacion concreta del servicio de tracking, preparada para streaming.
    - Mantiene self.last_tracked para acceso externo (p.ej. TeamAssigner)
    """

    def __init__(self, ball_model_path: str, player_model_path: str):
        super().__init__(ball_model_path=ball_model_path, player_model_path=player_model_path)
        self.last_tracked: sv.Detections | None = None

    def get_tracker(self, key: str):
        from app.infraestructure.trackers.tracker_factory import TrackerFactoryError

        try:
            tracker = self.tracker_factory.get_tracker(key)
            if not tracker:
                raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
            return tracker
        except Exception as e:
            logging.exception(f"Error getting tracker {key}: {e}")
            print(f"Error getting tracker {key}: {e}")
            raise e
