import json
import time
import tracemalloc
from typing import List

from app.modules.plotting import generate_diagrams
from app.modules.services import read_video
from app.modules.services.r2_download import R2Downloader
from app.modules.services.verify_model import prepare_model

from app.modules.trackers import TrackerService
from sqlalchemy.orm import Session

from app.tasks.analysis.process import process_frame
from app.tasks.analysis_tools import AnalysisTools
from app.tasks.upload_heatmaps import upload_heatmaps_for_extracted_players
from app.utils.routes import INPUT_VIDEOS_DIR, MODELS_DIR, OUTPUT_REPORTS_DIR
from app.logger import *

async def run_analysis(db: Session, video_name: str, match_id: int) -> None:
    try:
        export_data_file = OUTPUT_REPORTS_DIR / f"export_data_match_{match_id}.txt"
        export_data_file.parent.mkdir(parents=True, exist_ok=True)
        print("Archivo de reporteria creado: ", export_data_file.exists())
        
        if not export_data_file.exists():
            export_data_file.touch()
            info_logger.info("Volviendo a intentar: Archivo de reporteria creado en: " + export_data_file.as_posix())
    except Exception as e:
        print(f"Error creando archivo de reporteria: {e}")
        raise e
    # -----------------------------
    # MÉTRICAS Y CONFIGURACIÓN
    # -----------------------------
    tracemalloc.start()
    start_time = time.time()

    metrics = {
        "processing_time": [],
        "memory_usage": [],
        "ball_detection": {"detected": 0, "interpolated": 0},
        "interpolation_error": 0.0,
        "velocity_inconsistencies": {"players": 0, "referees": 0},
    }
    
    print("Prepara el modelo si es necesario...")
    model_path = MODELS_DIR / "football_model.torchscript"
    prepare_model(
        model_path=model_path,
        source_path=model_path.parent)
    
    # Descarga video
    downloader = R2Downloader()
    
    video_name = "fb64992c-0a84-4fb5-8c3c-42f4ddbfda1c-1_720p.mkv"

    print(f"Descargando video {video_name}...")
    download_path = downloader.build_destination_path(key=video_name, base_dir=INPUT_VIDEOS_DIR.as_posix())
    downloader.stream_download(key=video_name, destination_path=download_path.as_posix())
    print(f"Video descargado en {INPUT_VIDEOS_DIR.as_posix()}")
    print(f"Video descargado en {download_path.as_posix()}")

    # -----------------------------
    # LECTURA DEL VIDEO
    # -----------------------------
    video_stream = read_video(download_path.as_posix())
    images_per_player = 3
    if not video_stream:
        error_logger.error("Error: No frames read from video")
        return

    try:
        tracker = TrackerService(
            model_path.as_posix()
        )

        try:
            first_batch = next(video_stream)
            first_frame, _ = first_batch[0]
        except StopIteration:
            error_logger.error("Error: Video is empty")
            return

        if not first_frame.any():
            error_logger.error("Error: First frame is empty")
            return
        info_logger.info("Inicializando servicios de análisis...")
        tools = AnalysisTools()
        tools.start(db=db, first_frame=first_frame)
    except Exception as e:
        error_logger.error(f"Error initializing services: {e}")
        raise e

    frame_num = 0

    # ==========================================================================
    #                               LOOP PRINCIPAL
    # ==========================================================================
    empty_batches = 0
    max_empty_batches = 10
    max_processing_time = 1000 # 10 minutos por Debug
    saved_player_ids: List[int] = []
    for batch in video_stream:
        if not batch or len(batch) == 0:
            empty_batches += 1
            if empty_batches >= max_empty_batches:
                debug_logger.debug("Múltiples batches vacíos consecutivos, finalizando procesamiento.")
                break
            continue

        empty_batches = 0

        if time.time() - start_time > max_processing_time:
            debug_logger.debug("Tiempo de procesamiento excedido, finalizando.")
            break

        print(f"\n{'#'*60}\nProcesando batch de {len(batch)} frames...\n{'#'*60}\n")
        frame_num, updated_ids, updated_metrics = process_frame(
            video_batch=batch,
            frame_num=frame_num,
            db=db,
            tracker=tracker,
            tools=tools,
            metrics=metrics,
            images_per_player=images_per_player,
            saved_player_ids=saved_player_ids,
            export_data_file=export_data_file
        )
        saved_player_ids.extend(updated_ids)
        metrics.update(updated_metrics)
        info_logger.info(f"Batch procesado. Frames hasta ahora: {frame_num + len(batch)}")

    info_logger.info(f"Jugadores con imágenes extraídas {saved_player_ids}")

    generate_diagrams(db)
    info_logger.info("Diagramas generados.")
    await upload_heatmaps_for_extracted_players(db=db, match_id=match_id, extracted_player_ids=set(saved_player_ids))
    info_logger.info("Heatmaps subidos.")

    total_time = time.time() - start_time
    metrics.update(
        {
            "Frames por minuto": (frame_num / total_time) * 60,
            "Frames procesados": frame_num,
            "Tiempo total de procesamiento": f"{total_time/60:.2f} minutos"
            })

    print("\n" + "=" * 50)
    print("        RESUMEN FINAL DEL PROCESAMIENTO")
    print("=" * 50)
    print(f"Tiempo total: {total_time/60:.2f} min")
    # print(f"Memoria máxima usada: {max(metrics['memory_usage']):.2f} MB")
    info_logger.info(f"Frames balón detectado: {metrics['ball_detection']['detected']}")
    info_logger.info(f"Frames balón interpolado: {metrics['ball_detection']['interpolated']}")
    metrics_file = OUTPUT_REPORTS_DIR / f"metrics_match_{match_id}.json"
    info_logger.info(f"Escribiendo métricas a {metrics_file.as_posix()}...")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    info_logger.info("Métricas escritas.")



