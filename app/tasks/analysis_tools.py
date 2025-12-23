from sqlalchemy.orm import Session
from cv2.typing import MatLike

from app.modules.view_transformer.view_transformer import ViewTransformer
from app.modules.team_assigner.team_assigner import TeamAssigner
from app.modules.speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
from app.modules.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from app.modules.camera.camera_movement_estimator import CameraMovementEstimator
from app.entities.utils.singleton import Singleton
from app.entities.models import PlayerStateModel, BallEventModel
from app.entities.collections import TrackCollectionPlayer, TrackCollectionBall
from app.modules.services.video_processing_service import FPS_FRAME_RATE


class AnalysisTools(metaclass=Singleton):
    def __init__(self):
        pass

    def start(self, db: Session, first_frame: MatLike):
        self.player_records = TrackCollectionPlayer(db)
        self.player_records.orm_model = PlayerStateModel
        self.ball_records = TrackCollectionBall(db)
        self.ball_records.orm_model = BallEventModel
        self.view_transformer = ViewTransformer()
        self.speed_and_distance = SpeedAndDistanceEstimator(frame_rate=FPS_FRAME_RATE)
        self.team_assigner = TeamAssigner()
        self.player_ball_assigner = PlayerBallAssigner(fps=FPS_FRAME_RATE)
        self.camera_movement_estimator = CameraMovementEstimator(first_frame)

    def reset(self):
        pass
