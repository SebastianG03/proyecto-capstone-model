import logging
from typing import List, Tuple, Union

import supervision as sv
from cv2.typing import MatLike
import torch
from ultralytics.models import YOLO


from app.core.config import MODEL_USE_HALF_PRECISION
from app.entities.interfaces.tracker_base import Tracker
from app.entities.models.PlayerModels import Player, PlayerState
from app.entities.trackers import ball_tracker
from app.entities.trackers.player_tracker import PlayerTracker
from app.entities.utils.singleton import AbstractSingleton
from app.infraestructure.services.bbox_processor_service import get_center_of_bbox
from sqlalchemy.orm import Session


class TrackerServiceBase(metaclass=AbstractSingleton):
    """
    Servicio base para detección + tracking.
    - Carga detector (YOLO)
    - Mantiene un ByteTrack interno (self.tracker) para continuidad entre frames
    - Provee métodos streaming: process_frame (1 frame) y get_object_tracks
    """

    def __init__(self, ball_model_path: str, player_model_path: str):
        from app.infraestructure.trackers.tracker_factory import TrackerFactory

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ball_model = self.__load_detector__(ball_model_path)
        self.player_model = self.__load_detector__(player_model_path)

        self.tracker = sv.ByteTrack(
            frame_rate=25,
            lost_track_buffer=20,
            track_activation_threshold=0.5,
            minimum_matching_threshold=0.9,
            minimum_consecutive_frames=1,
        )
        self.tracker_factory = TrackerFactory(
            ball_model=self.ball_model,
            player_model=self.player_model
            )
        self.tracker_path = "bytetrack.yaml"
        logging.info(f"TrackerServiceBase initialized on device={self._device}")

    def __load_detector__(self, model_path: str) -> YOLO:
        try:
            model = YOLO(model_path, task="obb", verbose=False)
            
            if model_path.endswith(".pt"):
                model.fuse()

            if MODEL_USE_HALF_PRECISION:
                try:
                    model.half()
                except Exception as e:
                    logging.warning(f"Could not enable half precision: {e}")

            return model
        except Exception as e:
            logging.exception(f"Error loading model: {e}")
            raise e

    def __enter__(self):
        logging.info("Entering context: TrackerServiceBase")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if hasattr(self, "ball_model"):
                del self.ball_model
            if hasattr(self, "player_model"):
                del self.player_model
            if hasattr(self, "tracker"):
                del self.tracker

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logging.info("TrackerServiceBase resources released safely.")

        except Exception as e:
            logging.error(f"Error during context cleanup: {e}")

        # No suprime excepciones
        return False

    def reset_tracking(self) -> None:
        """
        Reinicia el tracker (p. ej. cuando se corta el stream o hay un gap grande)
        """
        self.tracker = sv.ByteTrack()
        logging.info("ByteTrack reset.")

    def detect_frames(
        self, frames: Union[List[MatLike], MatLike], conf: float = 0.1
    ) -> Tuple[List[sv.Detections], List[sv.Detections]]:
        """
        Detecta en una lista de frames o un solo frame.
        Retorna una tupla con dos listas de supervisions.Detections, correspondientes
        a las detecciones del modelo de pelota y del modelo de jugador, respectivamente.
        """

        print("Detectando en frames...")
        return self.ball_model(
            frames, conf=conf, iou=0.9, agnostic_nms=False, max_det=1000, nms=False
        ), self.player_model(
            frames, conf=conf, iou=0.9, agnostic_nms=False, max_det=1000, nms=False
        )

    def process_frame(
        self,
        frame: MatLike,
        frame_num: int,
        db: Session,
        conf: float = 0.1,
    ) -> None:
        """
        Procesa un único frame en modo streaming:
        1) detecta
        2) convierte a supervision
        3) actualiza ByteTrack
        4) distribuye a trackers registrados
        """
        try:
            # 1) Detectar (modelo retorna lista de Results aunque pasemos single frame)
            print("Detectando en frame...")
            ball_results, player_results = self.detect_frames([frame], conf=conf)
            print("Detección finalizada.")
            if not ball_results and not player_results:
                print("No se obtuvieron resultados de detección.")
                return

            # results[0] es la detección del frame
            print("Procesando resultados de detección...")
            player_result = player_results[0]
            ball_result = ball_results[0]

            # 2) map de clases
            print("Mapeando clases...")
            player_cls_names = getattr(player_result, "names", {})
            ball_cls_names = getattr(ball_result, "names", {})

            player_cls_names_inv = {v: k for k, v in player_cls_names.items()}
            ball_cls_names_inv = {v: k for k, v in ball_cls_names.items()}
            
            cls_names_inv = {**player_cls_names_inv, **ball_cls_names_inv}

            # 3) convertir a supervision
            player_detection = sv.Detections.from_ultralytics(player_result)
            ball_detection = sv.Detections.from_ultralytics(ball_result)

            # 4) tracking (ByteTrack) — devuelve detections con track_id
            print("Actualizando ByteTrack...")
            player_tracked = self.tracker.update_with_detections(player_detection)
            ball_tracked = self.tracker.update_with_detections(ball_detection)

            for tracker in self.get_trackers():
                try:
                    print(f"Ejecutando tracker {tracker}...")
                    if not tracker:
                        break

                    if not isinstance(tracker, Tracker):
                        print(
                            f"Tracker {tracker} no es instancia de Tracker. Se omite."
                        )
                        continue
                    is_player_tracker = isinstance(tracker, PlayerTracker)
                    tracker.get_object_tracks(
                        detection_with_tracks=player_tracked if is_player_tracker else ball_tracked,
                        cls_names_inv=cls_names_inv,
                        frame_num=frame_num,
                        detection_supervision=player_detection if is_player_tracker else ball_detection,
                        db=db,
                    )
                except Exception as e:
                    logging.exception(f"Error executing tracker {tracker}: {e}")
        except Exception as e:
            logging.exception(f"Error processing frame {frame_num}: {e}")

    def get_trackers(self) -> List:
        return list(self.tracker_factory.get_trackers().values())

    def add_position_to_track(
        self, db: Session, track
    ) -> None:
        from app.entities.models.BallState import BallEventModel
        
        try:
            bbox = track.get_bbox()
            print("Bbox del track ", track.id, ": ", bbox)

            if bbox is None or len(bbox) == 0:
                return

            print(
                f"Adding position to track {track.id} with bbox {bbox} de tipo {type(track)}"
            )
            position = get_center_of_bbox(bbox)
            if isinstance(track, Player):
                self.add_to_player(db, track, position)
            elif isinstance(track, BallEventModel):
                self.add_to_ball(db, track, position)
        except Exception as e:
            logging.exception(f"Error adding position to track {track}: {e}")
            print(f"Error adding position to track {track}: {e}")
            raise e

    def add_to_ball(
        self, db: Session, track, position: tuple[float, float]
    ) -> None:
        from app.entities.collections import TrackCollectionBall

        try:
            collection = TrackCollectionBall(db)
            collection.patch(int(f"{track.id}"), {"x": position[0], "y": position[1]})
        except Exception as e:
            logging.exception(f"Error adding to player {track}: {e}")
            print(f"Error adding to player {track}: {e}")
            raise e

    def add_to_player(
        self, db: Session, track: PlayerState, position: tuple[float, float]
    ) -> None:
        try:
            from app.entities.collections import TrackCollectionPlayer

            collection = TrackCollectionPlayer(db)
            collection.patch_state(
                frame_index=int(f"{track.frame_index}"),
                player_id=int(f"{track.player_id}"),
                updates={"x": position[0], "y": position[1]},
            )
        except Exception as e:
            logging.exception(f"Error adding to player {track}: {e}")
            print(f"Error adding to player {track}: {e}")
            raise e
