import logging

from sqlalchemy.orm import Session

from app.infraestructure.trackers.tracker_service import TrackerService
from app.entities.utils.tools_context import analysis_context
from app.logger import get_logger

logger = get_logger(logging.DEBUG)

def process_tracks_and_position(
    db: Session,
    tracker: TrackerService,
    camera_movement: tuple[float, float],
    pixels_to_meters: float,
):
    tools = analysis_context.tools
    logger.info("Procesando tracks y posiciones...")
    try:
        for collection in (tools.player_records, tools.ball_records):
            scale = tools.camera_movement_estimator.get_current_scale()

            last_track = collection.get_last()
            if last_track is None:
                logger.info("No hay track para actualizar, saltando...")
                continue
            logger.info(f"Último track ID: {last_track.to_dict().get('player_id', None)}")

            logger.info("Añadiendo posición al track...")
            tracker.add_position_to_track(db, last_track)

            logger.info("Ajustando posiciones según movimiento de cámara...")
            tools.camera_movement_estimator.add_adjust_positions_to_tracks(
                db=db,
                camera_movement_per_frame=camera_movement,
                pixels_to_meters=pixels_to_meters,
                scale=scale,
                track=last_track,
            )
            logger.info("Posiciones ajustadas.")

            logger.info("Aplicando transformación de vista...")
            tools.view_transformer.add_transformed_positions(db)
    except Exception as e:
        logger.error(f"Error procesando tracks y posiciones: {e}")
        raise e
