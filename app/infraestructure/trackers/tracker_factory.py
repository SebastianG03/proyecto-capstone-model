from typing import Dict, Type
from app.entities.interfaces.tracker_base import Tracker
from ultralytics.models import YOLO

from app.entities.utils.singleton import Singleton

class TrackerFactoryError(Exception):
    pass

class TrackerFactory(metaclass=Singleton):
    def __init__(self, model: YOLO):
        self.model = model
        # registramos clases (no instancias)
        self._trackers: Dict[str, Tracker] = {}
        # cache para instancias creadas (lazy)
        self._create_default_trackers()

    def _create_default_trackers(self) -> None:
        # Importar aquí para evitar circular imports al nivel module
        from app.entities.trackers.player_tracker import PlayerTracker
        from app.entities.trackers.ball_tracker import BallTracker

        self._trackers["player"] = PlayerTracker(self.model)
        self._trackers["ball"] = BallTracker(self.model)

    def _register_class(self, key: str, tracker_cls: Type[Tracker]) -> None:
        if key in self._trackers:
            raise TrackerFactoryError(f"Tracker '{key}' is already registered.")
        self._trackers[key] = tracker_cls(self.model)

    def get_trackers(self) -> Dict[str, Tracker]:
        return dict(self._trackers)

    def get_tracker(self, key: str) -> Tracker:
        if not key in self._trackers:
            raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
        tracker = self._trackers.get(key)
        if not tracker:
            raise TrackerFactoryError(f"Tracker '{key}' instance could not be created.")
        return tracker

    def reset_all(self) -> None:
        self._trackers = {}
