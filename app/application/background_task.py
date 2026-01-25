import json
import httpx
from datetime import datetime
from datetime import timezone

from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import Player, PlayerState
from app.infraestructure.services.database import create_temporary_database
from sqlalchemy.orm import Session

from app.application.post_process.proccess_final_data import analyze_match
from app.utils.routes import OUTPUT_REPORTS_DIR
from app.logger import error_logger, debug_logger, info_logger
from app.core.config import DEBUG, STATS_NOTIFY_URL
import traceback


async def process_video_async(video_name: str, match_id: int, color: str):
    """
    Ejecuta el análisis en segundo plano con una BD en memoria aislada.
    """
    info_logger.info(
        f"Iniciando análisis en background para video: {video_name}, match_id: {match_id}"
    )
    info_logger.info("Color enviado al análisis: " + color)
    db, engine, db_path = create_temporary_database(match_id)

    try:
        info_logger.info(f"Ejecutando análisis en background para video: {video_name}")
        _ = await process_run(
            db=db, video_name=video_name, match_id=match_id, color=color
        )
        info_logger.info("Análisis finalizado.")
    except Exception as e:
        error_logger.error(f"Error en análisis: {str(e)}")
    finally:
        info_logger.info("Cerrando sesión de base de datos y liberando recursos.")
        db.close()
        engine.dispose()
        if not DEBUG:
            db_path.unlink()


async def process_run(db: Session, video_name: str, match_id: int, color: str):
    try:
        start = datetime.now(timezone.utc)
        from app.application.analysis.runner import run_analysis

        info_logger.info("Analisis iniciado...")
        heatmaps = run_analysis(db=db, video_name=video_name, match_id=match_id)
        _ = await export_data(
            db, match_id, start_time=start, heatmaps=heatmaps or {}, color=color
        )
    except Exception as e:
        error_logger.error(f"[BACKGROUND_TASK] Error al analizar el video, error: {e}")
        error_logger.error(traceback.format_exc())
        raise e


async def export_data(
    db: Session,
    match_id: int,
    start_time: datetime,
    color: str,
    max_records: int = 100000,
    heatmaps: dict[int, str] = {},
):
    try:
        player_records = (
            db.query(PlayerState).order_by(PlayerState.id).limit(max_records).all()
        )
        players = db.query(Player).all()
        ball_records = (
            db
            .query(BallEventModel)
            .order_by(BallEventModel.id)
            .limit(max_records)
            .all()
        )
        print(f"Total de registros PlayerState a exportar: {len(player_records)}")
        print(f"Total de registros BallEvent a exportar: {len(ball_records)}")
        if not player_records or not ball_records:
            print("No hay registros de PlayerState para exportar.")
            return

        player_export_data = []
        ball_export_data = []

        player_stats = analyze_match(
            ball_events=ball_records,
            heatmaps=heatmaps,
            match_id=match_id,
            player_states=player_records,
            players=players,
            start_time=start_time,
        )

        for i, record in enumerate(ball_records):
            ball_export_data.append(record.to_dict())

        for i, record in enumerate(player_records):
            player = next(
                (p for p in players if f"{p.player_id}" == f"{record.player_id}"), None
            )
            if not player:
                continue
            dict_values = record.to_dict()
            dict_values.update({
                "team": player.team,
                "shirt_number": player.shirt_number,
                "color": player.color,
            })
            player_export_data.append(dict_values)
            if (i + 1) % 1000 == 0:
                print(f"Exportados {i + 1} registros de PlayerState...")
        print("Exportación de datos completada.")

        file_stats = OUTPUT_REPORTS_DIR / f"stats_match_{match_id}.json"
        file_stats.parent.mkdir(parents=True, exist_ok=True)
        file_stats.write_text(
            json.dumps(player_stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        ball_output_file = OUTPUT_REPORTS_DIR / f"ball_events_match_{match_id}.json"
        ball_output_file.parent.mkdir(parents=True, exist_ok=True)

        ball_output_file.write_text(
            json.dumps(ball_export_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        debug_logger.debug(
            f"Notificando emision de estadisticas para match_id {match_id}..."
        )
        resp = httpx.post(
            url=f"{STATS_NOTIFY_URL}/update-stats/",
            json={"match_id": match_id, "stats": player_stats, "color": color},
        )
        resp.raise_for_status()
        debug_logger.debug(
            f"Notificación enviada, respuesta: {resp.status_code} - {resp.text}"
        )
    except httpx.HTTPError as http_err:
        print(f"Error HTTP al exportar datos: {http_err}")
        raise http_err
    except Exception as e:
        error_logger.error(f"Error al exportar datos: {e}")
        raise e
