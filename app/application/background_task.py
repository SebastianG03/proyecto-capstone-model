from datetime import datetime, timezone
import json
import httpx
from sqlalchemy import event

from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import Player, PlayerState
from app.infraestructure.database.connection_manager import ConnectionManager
import app.entities.utils.global_values_store as value_store

from app.application.post_process.proccess_final_data import analyze_match
from app.utils.routes import BASE_RES_DIR, OUTPUT_REPORTS_DIR
from app.logger import error_logger, debug_logger, info_logger
from app.core.config import DEBUG, STATS_NOTIFY_URL
import traceback


async def process_video_async(video_name: str, match_id: int, color: str):
    """
    Ejecuta el analisis en segundo plano con una BD en memoria aislada.
    """
    info_logger.info(f"Iniciando analisis en background para video: {video_name}, match_id: {match_id}")
    db_path = BASE_RES_DIR / "database" / f"temp_db_{match_id}.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch(exist_ok=True)
    connection_manager = ConnectionManager(match_id=match_id)
    value_store.globals.connection_manager = connection_manager
    
    @event.listens_for(connection_manager.engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """Configura pragmas de SQLite para mejor rendimiento y recuperacion de bloqueos"""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.close()
    

    try:
        info_logger.info(f"Ejecutando analisis en background para video: {video_name}")
        _ = await process_run(video_name=video_name, match_id=match_id, color=color)
        info_logger.info("Analisis finalizado.")
    except Exception as e:
        error_logger.error(f"Error en analisis: {str(e)}")
    finally:
        info_logger.info("Cerrando sesion de base de datos y liberando recursos.")
        connection_manager.close_session()
        if not DEBUG:
            db_path.unlink()


async def process_run(video_name: str, match_id: int, color: str):
    try:
        start = datetime.now(timezone.utc)
        from app.application.analysis.runner import run_analysis

        info_logger.info("Analisis iniciado...")
        heatmaps = run_analysis(video_name=video_name, match_id=match_id)
        _ = await export_data(match_id, start_time=start, color=color)
    except Exception as e:
        error_logger.error(f"[BACKGROUND_TASK] Error al analizar el video, error: {e}")
        error_logger.error(traceback.format_exc())
        raise e


async def export_data(
    match_id: int,
    start_time: datetime,
    color: str,
    max_records: int = 100000,
):
    try:
        db = value_store.globals.connection_manager.create_session()
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
        print("Exportacion de datos completada.")

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
            f"Notificacion enviada, respuesta: {resp.status_code} - {resp.text}"
        )
    except httpx.HTTPError as http_err:
        print(f"Error HTTP al exportar datos: {http_err}")
        raise http_err
    except Exception as e:
        error_logger.error(f"Error al exportar datos: {e}")
        raise e
