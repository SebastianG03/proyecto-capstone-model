from datetime import datetime, timezone
import json
import pathlib
from sqlalchemy import create_engine

from app.entities.collections.track_collections import TrackCollectionPlayer
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerState import PlayerStateModel
from app.modules.services.database import Base
from sqlalchemy.orm import sessionmaker, Session

from app.tasks.upload import upload
from app.utils.routes import BASE_RES_DIR, OUTPUT_REPORTS_DIR

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
        from app.tasks.runner import run_analysis
        print("Analisis iniciado...")
        run_analysis(db=db, video_name=video_name, match_id=match_id)
        print("Exportando datos...")
        await export_data(db, match_id)
        print("Datos exportados.")

    except Exception as e:
        print(f"[ERROR procesando video]: {e}")

    
async def export_data(db: Session, match_id: int, max_records: int = 100000):
    try:
        player_records = (db.query(PlayerStateModel)
                .order_by(PlayerStateModel.id)
                .limit(max_records)
                .all())
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
        
        for i, record in enumerate(ball_records):
            ball_export_data.append(record.to_dict())
        
        for i, record in enumerate(player_records):
            player_export_data.append(record.to_dict())
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
        
        print("Subiendo datos exportados...")
        file_bytes = player_output_file.read_bytes()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        key = f"{match_id}/reports/{timestamp}_{player_output_file.name}"
        upload(
            key=key,
            file_bytes=file_bytes,
            file_type="application/json"
        )
        print("Datos subidos correctamente.")
    except Exception as e:
        print(f"Error al exportar datos: {e}")
        raise e
