import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from app.entities.models import PlayerState, BallEventModel
from app.shared.analysis_tools import AnalysisTools


def assign_ball_to_player(
    ball_records: List[BallEventModel],
    players: List[PlayerState],
    tools: AnalysisTools,
    dt: float,
    frame_index: int,
    db: Session,
    logger: logging.Logger
):
    try:
        logger.info("Iniciando asignación de balón a jugador...")
        logger.info(f"Colección de jugadores obtenida. Longitud actual: {len(players)}")
        logger.info(
            f"Iterando sobre registros de balón, registros actuales en colección: {len(ball_records)}"
        )
        for ball_track in ball_records:
            if not ball_track:
                logger.info("No hay balón en este frame, asignación por defecto.")
                continue

            logger.info("Obteniendo detalle del balón...")
            ball_bbox = ball_track.get_bbox()
            logger.info(
                f"Bbox del balón: {ball_bbox}, "
                f"numero de elementos en bbox: {len(ball_bbox) if ball_bbox else 'N/A'}"
            )
            logger.info(f"Bbox data type: {type(ball_bbox)}")
            if ball_bbox is None:
                logger.info("No hay balón en este frame, asignación por defecto.")
                continue

            logger.info("Asignando balón a jugador...")
            assigned_player_id = tools.player_ball_assigner.assign_ball_to_player(
                players=players,
                ball_event=ball_track,
                db=db,
                dt=dt,
                frame_number=frame_index,
                scale=tools.camera_movement_estimator.get_current_scale(),
            )
            logger.info(f"Jugador asignado ID (track_id): {assigned_player_id}")

            if assigned_player_id == -1:
                logger.info("No hay jugador asignado, asignación por defecto.")
                continue

            logger.info(f"Jugador asignado: {assigned_player_id}")
        logger.info("Asignación de balón completada.")
    except Exception as e:
        logger.info(f"Error en asignación de balón: {e}")
        raise e
