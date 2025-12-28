# tracker_service_base.py
from asyncio.log import logger
import logging
from typing import List, Union, TYPE_CHECKING

import supervision as sv
from cv2.typing import MatLike
import torch
from ultralytics.models import YOLO


from app.entities.collections import TrackCollectionBall, TrackCollectionPlayer
from app.entities.interfaces.tracker_base import Tracker
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import Player
from app.entities.utils.singleton import AbstractSingleton
from app.modules.services.bbox_processor_service import get_center_of_bbox
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from app.modules.trackers.tracker_factory import TrackerFactory


class TrackerServiceBase(metaclass=AbstractSingleton):
    """
    Servicio base para detección + tracking.
    - Carga detector (YOLO)
    - Mantiene un ByteTrack interno (self.tracker) para continuidad entre frames
    - Provee métodos streaming: process_frame (1 frame) y get_object_tracks (compatibilidad con listas)
    """

    def __init__(self, model_path: str, use_half_precision: bool = False):
        from app.modules.trackers.tracker_factory import TrackerFactory
        self.model = self.__load_detector__(model_path, use_half_precision)
        self.tracker = sv.ByteTrack(
            frame_rate=30,
            lost_track_buffer=60,
            track_activation_threshold=0.15,
            minimum_matching_threshold=0.9,
            minimum_consecutive_frames=3
        )
        self.tracker_factory = TrackerFactory(self.model)
        self.tracker_path = "bytetrack.yaml"
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"TrackerServiceBase initialized on device={self._device}")

    def __load_detector__(self, model_path: str, use_half_precision: bool = False) -> YOLO:
        model = YOLO(model=model_path, task='obb', verbose=False)
        if torch.cuda.is_available():
            model.to('cuda')
            if use_half_precision:
                try:
                    model.half()
                except Exception as e:
                    logging.warning(f"Could not enable half precision: {e}")
        return model

    def __del__(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared on TrackerServiceBase deletion.")
        except Exception as e:
            logging.error(f"Error during TrackerServiceBase deletion: {e}")

    def __enter__(self):
        logging.info("Entering context: TrackerServiceBase")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if hasattr(self, "model"):
                del self.model
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

    def detect_frames(self, frames: Union[List[MatLike], MatLike], conf: float = 0.1) -> list:
        """
        Wrapper sobre el detector. Acepta un frame o lista de frames y devuelve lista de Results.
        Nota: el modelo YOLO puede aceptar listas y devolver lista de Results.
        """
        print("Detectando en frames...")
        return self.model.predict(frames, conf=conf)

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
            results = self.detect_frames([frame], conf=conf)
            print("Detección finalizada.")
            if not results:
                print("No se obtuvieron resultados de detección.")
                return

            # results[0] es la detección del frame
            print("Procesando resultados de detección...")
            result = results[0]

            # 2) map de clases
            print("Mapeando clases...")
            cls_names = getattr(result, "names", {})
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # 3) convertir a supervision
            det_sv = sv.Detections.from_ultralytics(result)

            # 4) tracking (ByteTrack) — devuelve detections con track_id
            print("Actualizando ByteTrack...")
            tracked = self.tracker.update_with_detections(det_sv)

            for tracker in self.get_trackers():
                try:
                    print(f"Ejecutando tracker {tracker}...")
                    if not tracker:
                        break
                    
                    if not isinstance(tracker, Tracker):
                        print(f"Tracker {tracker} no es instancia de Tracker. Se omite.")
                        continue
                    
                    tracker.get_object_tracks(
                        detection_with_tracks=tracked,
                        cls_names_inv=cls_names_inv,
                        frame_num=frame_num,
                        detection_supervision=det_sv,
                        db=db
                    )
                except Exception as e:
                    logging.exception(f"Error executing tracker {tracker}: {e}")
        except Exception as e:
            logging.exception(f"Error processing frame {frame_num}: {e}")
            
    def get_trackers(self) -> List:
        return list(self.tracker_factory.get_trackers().values())

    def add_position_to_track(self, db: Session, track: Player | BallEventModel) -> None:
        try:
            bbox = track.get_bbox()
            print("Bbox del track ", track.id, ": ", bbox)

            if bbox is None or len(bbox) == 0:
                return

            print(f"Adding position to track {track.id} with bbox {bbox} de tipo {type(track)}")
            position = get_center_of_bbox(bbox)
            if isinstance(track, Player):
                self.add_to_player(db, track, position)
            elif isinstance(track, BallEventModel):
                self.add_to_ball(db, track, position)
        except Exception as e:
            logging.exception(f"Error adding position to track {track}: {e}")
            print(f"Error adding position to track {track}: {e}")
            raise e

    def add_to_ball(self, db: Session, track: BallEventModel, position: tuple[float, float]) -> None:
        try:
            collection = TrackCollectionBall(db)
            collection.patch(
                int(f'{track.id}'),
                {'x': position[0], 'y': position[1]})
        except Exception as e:
            logging.exception(f"Error adding to player {track}: {e}")
            print(f"Error adding to player {track}: {e}")
            raise e
    
    def add_to_player(self, db: Session, track: Player, position: tuple[float, float]) -> None:
        try:
            collection = TrackCollectionPlayer(db)
            collection.patch(
                int(f'{track.id}'),
                {'x': position[0], 'y': position[1]})
        except Exception as e:
            logging.exception(f"Error adding to player {track}: {e}")
            print(f"Error adding to player {track}: {e}")
            raise e
