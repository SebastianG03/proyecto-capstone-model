import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from app.core.config import BATCH_SIZE
from app.entities.models import PlayerState, BallEventModel
from app.shared.analysis_tools import AnalysisTools
import app.entities.utils.global_values_store as value_store

def assign_ball_to_player(
    ball_records: List[BallEventModel],
    players: List[PlayerState],
    tools: AnalysisTools,
    dt: float,
    depth: float,
    pixels_to_meters: float,
    frame_index: int,
    db: Session,
    logger: logging.Logger
):
    try:
        logger.info("[Assign Ball] Iniciando asignación de balón a jugador...")
        logger.info(f"[Assign Ball] Colección de jugadores obtenida. Longitud actual: {len(players)}")
        logger.info(
            f"[Assign Ball] Iterando sobre registros de balón, registros actuales en colección: {len(ball_records)}"
        )
        for ball_track in ball_records:
            if not ball_track:
                logger.info("[Assign Ball] No hay balón en este frame, asignación por defecto.")
                continue

            logger.info("[Assign Ball] Obteniendo detalle del balón...")
            ball_bbox = ball_track.get_bbox()
            if ball_bbox is None:
                logger.info("[Assign Ball] No hay balón en este frame, asignación por defecto.")
                continue

            logger.info("[Assign Ball] Asignando balón a jugador...")
            dt = dt * value_store.globals.fps if dt > 0 else BATCH_SIZE / value_store.globals.fps
            constant = tools.camera_movement_estimator.get_current_scale() * depth * pixels_to_meters
            logger.info(f"[Assign Ball] Parámetros para asignación - dt: {dt}, depth: {depth}, pixels_to_meters: {pixels_to_meters}, constant: {constant}")
            assigned_player_id = tools.player_ball_assigner.assign_ball_to_player(
                players=players,
                ball_event=ball_track,
                db=db,
                dt=dt,
                frame_number=frame_index,
                scale=constant,
            )
            logger.info(f"[Assign Ball] Jugador asignado ID (track_id): {assigned_player_id}")

            if assigned_player_id == -1:
                logger.info("[Assign Ball] No hay jugador asignado, asignación por defecto.")
                continue
    except Exception as e:
        raise e
