from sqlalchemy.orm import Session
from cv2.typing import MatLike

from app.entities.utils.singleton import Singleton


class AnalysisTools(metaclass=Singleton):
    
    def __init__(self):
        from app.entities.collections import TrackCollectionPlayer, TrackCollectionBall, TrackCollectionHeatmapPoint
        self.player_records: TrackCollectionPlayer
        self.ball_records: TrackCollectionBall
        self.heatmap_points: TrackCollectionHeatmapPoint
        from app.infraestructure.camera.number_recognizer import PlayerNumberDetector
        from app.infraestructure.camera.trocr_buffer import TROCRBuffer
        from app.infraestructure.view_transformer.view_transformer import ViewTransformer
        from app.infraestructure.team_assigner.team_assigner import TeamAssigner
        from app.infraestructure.speed_and_distance_estimator.speed_and_distance_estimator import (
            SpeedAndDistanceEstimator,
        )
        from app.infraestructure.player_ball_assigner.player_ball_assigner import (
            PlayerBallAssigner,
        )
        from app.infraestructure.camera.camera_movement_estimator import CameraMovementEstimator
        
        self.view_transformer: ViewTransformer
        self.speed_and_distance: SpeedAndDistanceEstimator
        self.team_assigner: TeamAssigner
        self.player_ball_assigner: PlayerBallAssigner
        self.camera_movement_estimator: CameraMovementEstimator
        self.number_recognizer: PlayerNumberDetector
        self.trocr_buffer: TROCRBuffer

    def start(self, db: Session, first_frame: MatLike):
        from app.entities.utils.tools_context import analysis_context
        from app.entities.models import PlayerState, BallEventModel
        from app.entities.collections import TrackCollectionPlayer, TrackCollectionBall, TrackCollectionHeatmapPoint
        from app.entities.utils.global_values_store import GlobalValuesStore
        from app.infraestructure.camera.number_recognizer import PlayerNumberDetector
        from app.infraestructure.camera.trocr_buffer import TROCRBuffer
        from app.infraestructure.view_transformer.view_transformer import ViewTransformer
        from app.infraestructure.team_assigner.team_assigner import TeamAssigner
        from app.entities.models import HeatmapPoint
        from app.infraestructure.speed_and_distance_estimator.speed_and_distance_estimator import (
            SpeedAndDistanceEstimator,
        )
        from app.infraestructure.player_ball_assigner.player_ball_assigner import (
            PlayerBallAssigner,
        )
        from app.infraestructure.camera.camera_movement_estimator import CameraMovementEstimator
        from app.utils.routes import TROCR_PATH

        globals = GlobalValuesStore()
        self.player_records = TrackCollectionPlayer(db)
        self.player_records.orm_model = PlayerState
        self.ball_records = TrackCollectionBall(db)
        self.ball_records.orm_model = BallEventModel
        self.heatmap_points = TrackCollectionHeatmapPoint(db)
        self.heatmap_points.orm_model = HeatmapPoint
        
        self.view_transformer = ViewTransformer()
        self.speed_and_distance = SpeedAndDistanceEstimator(frame_rate=globals.fps)
        self.team_assigner = TeamAssigner()
        self.player_ball_assigner = PlayerBallAssigner(fps=globals.fps)
        self.camera_movement_estimator = CameraMovementEstimator(first_frame)
        self.number_recognizer = PlayerNumberDetector(TROCR_PATH.as_posix())
        self.trocr_buffer = TROCRBuffer()
        analysis_context.tools = self

    def reset(self):
        pass
