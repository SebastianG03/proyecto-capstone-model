from typing import Dict, Type
from app.entities.interfaces.tracker_base import Tracker
from ultralytics.models import YOLO

from app.entities.trackers.player_tracker import PlayerTracker
from app.entities.utils.singleton import Singleton


class TrackerFactoryError(Exception):
    pass


class TrackerFactory(metaclass=Singleton):
    def __init__(self, ball_model: YOLO, player_model: YOLO):
        self.ball_model = ball_model
        self.player_model = player_model
        # registramos clases (no instancias)
        self._trackers: Dict[str, Tracker] = {}
        # cache para instancias creadas (lazy)
        self._create_default_trackers()
        

    def _create_default_trackers(self) -> None:
        # Importar aquí para evitar circular imports al nivel module
        from app.entities.trackers.player_tracker import PlayerTracker
        from app.entities.trackers.ball_tracker import BallTracker

        self._trackers["player"] = PlayerTracker(self.player_model)
        self._trackers["ball"] = BallTracker(self.ball_model)

    def _register_class(self, key: str, tracker_cls: Type[Tracker]) -> None:
        from app.entities.trackers.player_tracker import PlayerTracker
        from app.entities.trackers.ball_tracker import BallTracker
        if key in self._trackers:
            raise TrackerFactoryError(f"Tracker '{key}' is already registered.")
        if issubclass(tracker_cls, PlayerTracker):
            self._trackers[key] = tracker_cls(self.player_model)
        elif issubclass(tracker_cls, BallTracker):
            self._trackers[key] = tracker_cls(self.ball_model)

    def get_trackers(self) -> Dict[str, Tracker]:
        return dict(self._trackers)

    def get_tracker(self, key: str) -> Tracker:
        if key not in self._trackers:
            raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
        tracker = self._trackers.get(key)
        if not tracker:
            raise TrackerFactoryError(f"Tracker '{key}' instance could not be created.")
        return tracker

    def reset_all(self) -> None:
        self._trackers = {}
        self._create_default_trackers()