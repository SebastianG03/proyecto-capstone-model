import supervision as sv
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from sqlalchemy.orm import Session

class Tracker():
    """
    Interfaz base para trackers específicos (players, ball, referees, etc).
    - NO deben ejecutar el detector.
    - NO deben crear su propio ByteTrack.
    - Deben implementar get_object_tracks que recibe detecciones ya trackeadas.
    """

    def __init__(self, model):
        self.model = model
    
    def _bbox_to_center(self, bbox: list) -> tuple[float, float]:
        print("[Tracker] Convirtiendo bbox a centro...")
        x1, y1, x2, y2 = bbox
        print(f"[Tracker] Bbox recibida: {bbox}, coordenadas extraídas: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        cx = float((x1 + x2) / 2.0)
        print(f"[Tracker] cx={cx}")
        cy = float((y1 + y2) / 2.0)
        print(f"[Tracker] cy={cy}")
        return cx, cy

    def get_object_tracks(
            self,
            detection_with_tracks: sv.Detections,
            cls_names_inv: dict[str, int],
            frame_num: int,
            detection_supervision: sv.Detections,
            db: Session) -> None:
        """
        Procesa detecciones *ya trackeadas* por el servicio:
        - detection_with_tracks: sv.Detections con atributos de tracking (id, etc.)
        - cls_names_inv: mapping id->nombre de clase para interpretar label indices
        - frame_num: numero de frame relativo al batch/frame procesado
        - detection_supervision: detecciones originales en formato supervision (sin tracks)
        - tracks_collection: repo/collection para persistir resultados
        """
        print("Tracker.get_object_tracks called.")
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resetea el estado interno del tracker.
        NO toca la DB.
        """
        raise NotImplementedError
