import torch
import logging
from cv2.typing import MatLike

import app.entities.collections as collections
import app.infraestructure.camera as camera
import app.infraestructure.view_transformer as visual_transformer
import app.infraestructure.team_assigner as team_estimator
import app.infraestructure.speed_and_distance_estimator as movement_estimator
import app.infraestructure.player_ball_assigner as ball_assigner
import app.entities.models as models
import app.entities.utils.global_values_store as value_store
import app.entities.utils.tools_context as context
import app.logger as logger_lib

from app.entities.utils.singleton import Singleton
from app.utils.routes import TROCR_PATH


class AnalysisTools(metaclass=Singleton):
    
    def __init__(self):
        self.player_records: collections.TrackCollectionPlayer
        self.ball_records: collections.TrackCollectionBall
        self.heatmap_points: collections.TrackCollectionHeatmapPoint
        self.view_transformer: visual_transformer.ViewTransformer
        self.speed_and_distance: movement_estimator.SpeedAndDistanceEstimator
        self.team_assigner: team_estimator.TeamAssigner
        self.player_ball_assigner: ball_assigner.PlayerBallAssigner
        self.camera_movement_estimator: camera.CameraMovementEstimator
        self.number_recognizer: camera.PlayerNumberDetector
        self.trocr_buffer: camera.TROCRBuffer
        self.depth_estimator: camera.DepthEstimator
        self.analysis_data_collector: collections.DetectedDataAnalysis

    def start(self, first_frame: MatLike, match_id: int):
        """
        Initialize all the necessary tools for the analysis

        Parameters
        ----------
        db : Session
            The database session
        first_frame : MatLike
            The first frame of the video

        Notes
        -----
        This function initializes all the necessary tools for the analysis.
        It creates instances of the TrackCollection classes for the player, ball, and heatmap points,
        and initializes the ViewTransformer, SpeedAndDistanceEstimator, TeamAssigner, PlayerBallAssigner,
        CameraMovementEstimator, PlayerNumberDetector, TROCRBuffer, and DepthEstimator classes.
        """

        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger = logger_lib.get_logger(logging.DEBUG)

        self.player_records = collections.TrackCollectionPlayer()
        self.player_records.orm_model = models.PlayerState
        self.ball_records = collections.TrackCollectionBall()
        self.ball_records.orm_model = models.BallEventModel
        self.heatmap_points = collections.TrackCollectionHeatmapPoint()
        self.heatmap_points.orm_model = models.HeatmapPoint
        
        self.view_transformer = visual_transformer.ViewTransformer()
        fps = value_store.globals.fps
        self.speed_and_distance = movement_estimator.SpeedAndDistanceEstimator(frame_rate=fps)
        self.team_assigner = team_estimator.TeamAssigner()
        self.player_ball_assigner = ball_assigner.PlayerBallAssigner(fps=fps)
        self.camera_movement_estimator = camera.CameraMovementEstimator(first_frame, logger)
        self.number_recognizer = camera.PlayerNumberDetector(TROCR_PATH.as_posix())
        self.trocr_buffer = camera.TROCRBuffer()
        self.depth_estimator = camera.DepthEstimator(device, frame_rate=fps)
        self.analysis_data_collector = collections.DetectedDataAnalysis(match_id=match_id)
        
        context.analysis_context.tools = self

    def reset(self):
        pass
