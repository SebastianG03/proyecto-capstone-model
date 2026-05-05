import csv
from pathlib import Path
from typing import Optional

from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import Session

from app.entities.models.detected_object_data import AnalysisData
from app.utils.routes import DETECTED_OBJECTS_METRICS_DIR
from app.logger.logger import info_logger, error_logger
import app.entities.utils.global_values_store as value_store

class DetectedDataAnalysis:
    HEADERS = [
        "frame",
        "track_id",
        "x",
        "y",
        "vclass",
        "shirt_number",
        "velocity",
        "timestamps",
    ]

    def __init__(self, match_id: int):
        self.match_id = match_id
        self.csv_file = Path(
            DETECTED_OBJECTS_METRICS_DIR,
            f"detected_objects_{match_id}.csv",
        )

    def _add_row(self, data: AnalysisData, db: Session):
        obj = AnalysisData(
            frame=data.frame,
            track_id=data.track_id,
            x=data.x,
            y=data.y,
            vclass=data.vclass,
            shirt_number=data.shirt_number,
            velocity=data.velocity,
            timestamps=data.timestamps,
        )
        db.add(obj)
        
    def add_row(self, data: AnalysisData):
        try:
            db = value_store.globals.session
            self._add_row(data, db)
        except InvalidRequestError as ie:
            db = value_store.globals.connection_manager.create_session()
            value_store.globals.session = db
            self._add_row(data, db)
        except Exception as e:
            error_logger.error(f"[AnalysisDBError] Error al agregar registro: {e}")

    def _update_row(
        self,
        frame: int,
        track_id: int,
        db: Session,
        shirt_number: Optional[int] = None,
        velocity: Optional[float] = None,
    ):
        record = (db.query(AnalysisData)
                 .filter(AnalysisData.frame == frame)
                 .filter(AnalysisData.track_id == track_id)
                 .first())

        if not record:
            return

        if shirt_number is not None:
            setattr(record, "shirt_number", shirt_number)

        if velocity is not None:
            setattr(record, "velocity", velocity)

        db.flush()
        db.commit()
        db.refresh(record)
    
    def update_row(
        self,
        frame: int,
        track_id: int,
        shirt_number: Optional[int] = None,
        velocity: Optional[float] = None,
    ):
        try:
            db = value_store.globals.session
            self._update_row(frame, track_id, db, shirt_number, velocity)
        except InvalidRequestError as ie:
            db = value_store.globals.connection_manager.create_session()
            value_store.globals.session = db
            self._update_row(frame, track_id, db, shirt_number, velocity)
        except Exception as e:
            error_logger.error(f"[AnalysisDBError] Error al actualizar registro: {e}")


    def export_to_csv(self):
        info_logger.info("[DetectedDataAnalysis] Exporting CSV")
        db = value_store.globals.session
        data = (db
                .query(AnalysisData)
                .order_by(AnalysisData.frame, AnalysisData.track_id)
                .all())
        
        file = open(self.csv_file, "w", newline="", encoding="utf-8")
        writer = csv.writer(file)
        writer.writerow(self.HEADERS)
        
        for object in data:
            row = object.to_dict().values()
            writer.writerow(row)

        info_logger.info(f"CSV exported -> {self.csv_file}")
