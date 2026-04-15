import logging
from typing import List

import supervision as sv
from ultralytics.models import YOLO
from sqlalchemy.orm import Session
from app.logger import get_logger


class Tracker:
    """
    Interfaz base para trackers específicos (players, ball, referees, etc).
    - NO deben ejecutar el detector.
    - NO deben crear su propio ByteTrack.
    - Deben implementar get_object_tracks que recibe detecciones ya trackeadas.
    """

    def __init__(self, model: YOLO):
        self.model: YOLO = model
        self.logger = get_logger(logging.DEBUG)
        self.tracker = sv.ByteTrack(
            frame_rate=30,
            lost_track_buffer=60,
            track_activation_threshold=0.15,
            minimum_matching_threshold=0.9,
            minimum_consecutive_frames=1
        )

    def _bbox_to_center(self, bbox: list) -> tuple[float, float]:
        self.logger.info("[Tracker] Convirtiendo bbox a centro...")
        x1, y1, x2, y2 = bbox
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        return cx, cy

    def get_object_tracks(
        self,
        detections: List,
        cls_names_inv: dict[str, int],
        frame_num: int,
        db: Session,
    ) -> None:
        """
        Procesa detecciones *ya trackeadas* por el servicio:
        - detection_with_tracks: sv.Detections con atributos de tracking (id, etc.)
        - cls_names_inv: mapping id->nombre de clase para interpretar label indices
        - frame_num: numero de frame relativo al batch/frame procesado
        - detection_supervision: detecciones originales en formato supervision (sin tracks)
        - tracks_collection: repo/collection para persistir resultados
        """
        self.logger.info("Tracker.get_object_tracks called.")
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resetea el estado interno del tracker.
        NO toca la DB.
        """
        raise NotImplementedError
