from sqlalchemy.orm import Session
from cv2.typing import MatLike

from app.entities.utils.global_values_store import GlobalValuesStore
from app.infraestructure.camera.number_recognizer import PlayerNumberNP
from app.infraestructure.view_transformer.view_transformer import ViewTransformer
from app.infraestructure.team_assigner.team_assigner import TeamAssigner
from app.infraestructure.speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
from app.infraestructure.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from app.infraestructure.camera.camera_movement_estimator import CameraMovementEstimator
from app.entities.utils.singleton import Singleton
from app.entities.models import PlayerState, BallEventModel
from app.entities.collections import TrackCollectionPlayer, TrackCollectionBall


class AnalysisTools(metaclass=Singleton):
    def __init__(self):
        pass

    def start(self, db: Session, first_frame: MatLike):
        globals = GlobalValuesStore()
        self.player_records = TrackCollectionPlayer(db)
        self.player_records.orm_model = PlayerState
        self.ball_records = TrackCollectionBall(db)
        self.ball_records.orm_model = BallEventModel
        self.view_transformer = ViewTransformer()
        self.speed_and_distance = SpeedAndDistanceEstimator(frame_rate=globals.fps)
        self.team_assigner = TeamAssigner()
        self.player_ball_assigner = PlayerBallAssigner(fps=globals.fps)
        self.camera_movement_estimator = CameraMovementEstimator(first_frame)
        self.number_recognizer = PlayerNumberNP()

    def reset(self):
        pass
