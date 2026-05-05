import logging
from typing import List, Tuple, Union

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import supervision as sv
from cv2.typing import MatLike
import torch
from ultralytics.models import YOLO


from app.core.config import BATCH_SIZE, MODEL_USE_HALF_PRECISION
from app.entities.interfaces.tracker_base import Tracker
from app.entities.models.PlayerModels import Player, PlayerState
from app.entities.trackers.player_tracker import PlayerTracker
from app.entities.utils.singleton import AbstractSingleton
from app.infraestructure.services.bbox_processor_service import get_center_of_bbox
from sqlalchemy.orm import Session

from app.logger import get_logger

import app.entities.utils.global_values_store as value_store
import app.utils.routes as routes

class TrackerServiceBase(metaclass=AbstractSingleton):
    """
    Servicio base para deteccion + tracking.
    - Carga detector (YOLO)
    - Mantiene un ByteTrack interno (self.tracker) para continuidad entre frames
    - Provee metodos streaming: process_frame (1 frame) y get_object_tracks
    """

    def __init__(self, ball_model_path: str, player_model_path: str):
        from app.infraestructure.trackers.tracker_factory import TrackerFactory

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ball_model = self.__load_detector__(ball_model_path)
        self.player_model = self.__load_detector__(player_model_path)
        self.logger = get_logger(logging.DEBUG)

        self.ball_tracker = sv.ByteTrack(
            frame_rate=25,
            lost_track_buffer=8,
            track_activation_threshold=0.15,
            minimum_matching_threshold=0.5,
            minimum_consecutive_frames=3,
        )
        self.player_tracker = sv.ByteTrack(
            frame_rate=int(value_store.globals.fps),
            lost_track_buffer=50000,
            track_activation_threshold=0.4,
            minimum_matching_threshold=0.5,
            minimum_consecutive_frames=8,
        )
        self.tracker_factory = TrackerFactory(
            ball_model=self.ball_model,
            player_model=self.player_model
            )
        self.tracker_path = "bytetrack.yaml"
        self.logger.info(f"TrackerServiceBase initialized on device={self._device}")

    def __load_detector__(self, model_path: str) -> YOLO:
        try:
            model = YOLO(model_path, task="obb", verbose=False)
            
            if model_path.endswith(".pt"):
                model.fuse()

            if MODEL_USE_HALF_PRECISION:
                try:
                    model.half()
                except Exception as e:
                    self.logger.warning(f"Could not enable half precision: {e}")

            return model
        except Exception as e:
            self.logger.exception(f"Error loading model: {e}")
            raise e

    def __enter__(self):
        self.logger.info("Entering context: TrackerServiceBase")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if hasattr(self, "ball_model"):
                del self.ball_model
            if hasattr(self, "player_model"):
                del self.player_model
            

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("TrackerServiceBase resources released safely.")

        except Exception as e:
            self.logger.error(f"Error during context cleanup: {e}")

        # No suprime excepciones
        return False

    def apply_dbscan_to_detections(
    self,
    detections: sv.Detections,
    eps: float = 30,
    min_samples: int = 1
) -> sv.Detections:
        """
        Agrupa detecciones por cercania espacial usando DBSCAN.
        - eps: distancia maxima entre puntos para considerarse vecinos
        - min_samples: minimo para formar cluster
        """

        if len(detections) == 0:
            return detections

        centers = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = bbox
            centers.append([(x1 + x2) / 2, (y1 + y2) / 2])

        centers = np.array(centers)

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
        labels = clustering.labels_

        unique_indices = []

        for label in set(labels):
            cluster_indices = np.where(labels == label)[0]

            # Estrategia: elegir bbox con mayor area o confianza
            best_idx = cluster_indices[0]
            unique_indices.append(best_idx)

        # Filtrar detecciones
        filtered = detections[unique_indices]

        return filtered

    def reset_tracking(self) -> None:
        """
        Reinicia el tracker (p. ej. cuando se corta el stream o hay un gap grande)
        """
        self.ball_tracker.reset()
        self.player_tracker.reset()
        self.logger.info("ByteTrack reset.")

    def detect_frames(
        self, frame: MatLike
    ):
        """
        Detecta en una lista de frames o un solo frame.
        Retorna una tupla con dos listas de supervisions.Detections, correspondientes
        a las detecciones del modelo de pelota y del modelo de jugador, respectivamente.
        """

        print("Detectando en frames...")
        return self.ball_model(
            frame,
            imgsz=640,
            conf=0.3,
            iou=0.5,
            agnostic_nms=True,
            device=self._device,
            max_det=20,
            verbose=False,
        ), self.player_model.track(
            frame,
            imgsz=640,
            conf=0.15,
            iou=0.7,
            device=self._device,
            agnostic_nms=False,
            max_det=50,
            verbose=False,
            tracker=routes.BYTETRACK_CONFIG_PATH.as_posix(),
            persist=True
        )

    def get_object_tracks(
        self,
        frame: MatLike,
        frame_num: int,
        db: Session,
        conf: float = 0.1,
    ) -> None:
        """
        Procesa un unico frame en modo streaming:
        1) detecta
        2) convierte a supervision
        3) actualiza ByteTrack
        4) distribuye a trackers registrados
        """
        try:
            self.logger.info("Detectando en frame...")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ball_results, player_results = self.detect_frames(frame)

            if not ball_results and not player_results:
                self.logger.info("No se obtuvieron resultados de deteccion.")
                return

            self.logger.info("Procesando resultados de deteccion...")
            player_result = player_results[0]
            ball_result = ball_results[0]
          
            self.logger.info("Mapeando clases...")
            player_cls_names = getattr(player_result, "names", {})
            ball_cls_names = getattr(ball_result, "names", {})

            player_cls_names_inv = {v: k for k, v in player_cls_names.items()}
            ball_cls_names_inv = {v: k for k, v in ball_cls_names.items()}
            
            # player_detection = sv.Detections(xyxy=player_result.xyxy, confidence=player_result.confidence, class_id=np.zeros(len(player_result), dtype=int))
            # ball_detection = sv.Detections(xyxy=ball_result.xyxy, confidence=ball_result.confidence, class_id=np.zeros(len(ball_result), dtype=int))

            # self.logger.info("Actualizando ByteTrack...")
            # player_tracked = self.player_tracker.update_with_detections(player_detection)
            # ball_tracked = self.ball_tracker.update_with_detections(ball_detection)

            for tracker in self.get_trackers():
                try:
                    self.logger.info(f"Ejecutando tracker {tracker}...")
                    if not tracker:
                        break

                    if not isinstance(tracker, Tracker):
                        self.logger.info(
                            f"Tracker {tracker} no es instancia de Tracker. Se omite."
                        )
                        continue
                    is_player_tracker = isinstance(tracker, PlayerTracker)
                    if is_player_tracker:    
                        tracker.get_object_tracks(
                            detections=player_results,
                            cls_names_inv=player_cls_names_inv,
                            frame_num=frame_num,
                            db=db,
                            frame=frame
                        )
                    else:
                        tracker.get_object_tracks(
                            detections=ball_results,
                            cls_names_inv=ball_cls_names_inv,
                            frame_num=frame_num,
                            db=db,
                            frame=frame
                        )
                except Exception as e:
                    self.logger.exception(f"Error executing tracker {tracker}: {e}")
        except Exception as e:
            self.logger.exception(f"Error processing frame {frame_num}: {e}")

    def get_trackers(self) -> List:
        return list(self.tracker_factory.get_trackers().values())

    def add_position_to_track(
        self, db: Session, track
    ) -> None:
        from app.entities.models.BallState import BallEventModel
        
        try:
            bbox = track.get_bbox()
            self.logger.info(f"Bbox del track ${track.id}: ${bbox}")

            if bbox is None or len(bbox) == 0:
                return

            self.logger.info(
                f"Adding position to track {track.id} with bbox {bbox} de tipo {type(track)}"
            )
            position = get_center_of_bbox(bbox)
            if isinstance(track, Player):
                self.add_to_player(db, track, position)
            elif isinstance(track, BallEventModel):
                self.add_to_ball(db, track, position)
        except Exception as e:
            self.logger.exception(f"Error adding position to track {track}: {e}")
            print(f"Error adding position to track {track}: {e}")
            raise e

    def add_to_ball(
        self, db: Session, track, position: tuple[float, float]
    ) -> None:
        from app.entities.collections import TrackCollectionBall

        try:
            collection = TrackCollectionBall()
            collection.patch(int(f"{track.id}"), {"x": position[0], "y": position[1]})
        except Exception as e:
            self.logger.exception(f"Error adding to player {track}: {e}")
            raise e

    def add_to_player(
        self, db: Session, track: PlayerState, position: tuple[float, float]
    ) -> None:
        try:
            from app.entities.collections import TrackCollectionPlayer

            collection = TrackCollectionPlayer()
            
            collection.patch_state(
                frame_index=int(f"{track.frame_index}"),
                player_id=int(f"{track.player_id}"),
                updates={"x": position[0], "y": position[1]},
            )
        except Exception as e:
            self.logger.exception(f"Error adding to player {track}: {e}")
            raise e
