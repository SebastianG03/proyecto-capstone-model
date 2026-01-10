import json
import httpx
from sqlalchemy import create_engine

from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import Player, PlayerState
from app.infraestructure.services.database import Base
from sqlalchemy.orm import sessionmaker, Session

from app.application.post_process.proccess_final_data import analyze_match
from app.utils.routes import BASE_RES_DIR, OUTPUT_REPORTS_DIR
from app.logger import error_logger

async def process_video_async(video_name: str, match_id: int):
    """
    Ejecuta el análisis en segundo plano con una BD en memoria aislada.
    """
    print(f"Iniciando análisis en background para video: {video_name}, match_id: {match_id}")
    db_path = BASE_RES_DIR / "database" / f"temp_db_{match_id}.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path.as_posix()}", echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        print(f"Ejecutando análisis en background para video: {video_name}")
        await process_run(
            db=db,
            video_name=video_name,
            match_id=match_id,
            db_session_factory=SessionLocal)
        print("Análisis finalizado.")
    except Exception as e:
        print(f"Error en análisis: {str(e)}")
    finally:
        print("Cerrando sesión de base de datos y liberando recursos.")
        db.close()
        engine.dispose()
        # db_path.unlink(missing_ok=True)


async def process_run(db: Session, video_name: str, match_id: int, db_session_factory):
    try:
        from app.application.analysis.runner import run_analysis
        print("Analisis iniciado...")
        heatmaps = run_analysis(db=db, video_name=video_name, match_id=match_id)
        await export_data(db, match_id, heatmaps=heatmaps)
    except Exception as e:
        error_logger.error(f"[BACKGROUND_TASK] Error al analizar el video, error: {e.with_traceback(e.__traceback__)}")

async def export_data(db: Session, match_id: int, max_records: int = 100000, heatmaps: dict[int, str] | None = None):
    try:
        player_records = (db.query(PlayerState)
                .order_by(PlayerState.id)
                .limit(max_records)
                .all())
        players = (db.query(Player).all())
        ball_records = (db.query(BallEventModel)
                .order_by(BallEventModel.id)
                .limit(max_records)
                .all())
        print(f"Total de registros PlayerState a exportar: {len(player_records)}")
        print(f"Total de registros BallEvent a exportar: {len(ball_records)}")
        if not player_records or not ball_records:
            print("No hay registros de PlayerState para exportar.")
            return

        player_export_data = []
        ball_export_data = []
        
        player_stats = analyze_match(player_records, ball_records, players, heatmaps)
        
        for i, record in enumerate(ball_records):
            ball_export_data.append(record.to_dict())
        
        for i, record in enumerate(player_records):
            player = next((p for p in players if f'{p.player_id}' == f'{record.player_id}'), None)
            if not player:
                continue
            dict_values = record.to_dict()
            dict_values.update({"team": player.team, "shirt_number": player.shirt_number, "color": player.color})
            player_export_data.append(dict_values)
            if (i + 1 ) % 1000 == 0:
                print(f"Exportados {i + 1} registros de PlayerState...")
        print("Exportación de datos completada.")


        player_output_file = OUTPUT_REPORTS_DIR / f"player_states_match_{match_id}.json"
        ball_output_file = OUTPUT_REPORTS_DIR / f"ball_events_match_{match_id}.json"
        player_output_file.parent.mkdir(parents=True, exist_ok=True)
        ball_output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exportando datos a {player_output_file}")
        
        player_output_file.write_text(json.dumps(player_export_data, indent=2, ensure_ascii=False), encoding="utf-8")
        ball_output_file.write_text(json.dumps(ball_export_data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print("Subiendo datos a stats service..")
        httpx.post(
            url="http://127.0.0.1:8090/api/upload_consolidated/",
            json=player_stats)
    except Exception as e:
        print(f"Error al exportar datos: {e}")
        raise e
