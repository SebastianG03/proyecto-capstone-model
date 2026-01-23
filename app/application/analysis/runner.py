import json
from pathlib import Path
import time
import tracemalloc
from typing import List

from app.core.config import MAX_EMPTY_BATCHES, MAX_PROCESSING_TIME
from app.infraestructure.plotting import generate_diagrams

from app.infraestructure.trackers import TrackerService
from sqlalchemy.orm import Session

from app.application.analysis.process import process_frame
from app.shared.analysis_tools import AnalysisTools
from app.infraestructure.services import (
    upload_heatmaps_for_extracted_players, upload,
    R2Downloader, prepare_model, read_video
    )
import traceback

from app.utils.routes import INPUT_VIDEOS_DIR, MODEL_PATH, MODELS_DIR, OUTPUT_REPORTS_DIR
from app.logger import debug_logger, error_logger, info_logger

def run_analysis(db: Session, video_name: str, match_id: int) -> dict[int, str] | None:
    try:
        export_data_file = OUTPUT_REPORTS_DIR / f"export_data_match_{match_id}.txt"
        export_data_file.parent.mkdir(parents=True, exist_ok=True)
        export_data_file.touch(exist_ok=True)
        print("Archivo de reporteria creado: ", export_data_file.exists())        
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
    prepare_model(
        model_path=MODEL_PATH,
        source_path=MODEL_PATH.parent)
    
    # Descarga video
    downloader = R2Downloader()
    
    #video_name = "cc6dcc09-b6ed-41ad-8a83-3532ae0e11cc-VID_20260105_211836.mp4"

    print(f"Descargando video {video_name}...")
    download_path = Path(INPUT_VIDEOS_DIR, video_name)
    downloader.build_destination_path(key=video_name, base_dir=INPUT_VIDEOS_DIR.as_posix())
    downloader.stream_download(key=video_name, destination_path=download_path.as_posix())
    #download_path = Path("C:\\Users\\DavidGuaman\\Desktop\\universidad-capstone\\IAAnalisisModel\\app\\res\\input_videos\\9.mp4")
    print(f"Video descargado en {INPUT_VIDEOS_DIR.as_posix()}")
    print(f"Video descargado en {download_path.as_posix()}")
    # -----------------------------
    # LECTURA DEL VIDEO
    # -----------------------------
    print(download_path)
    print(type(download_path))
    video_stream = read_video(download_path.as_posix())
    images_per_player = 3
    if not video_stream:
        error_logger.error("Error: No frames read from video")
        return

    try:
        tracker = TrackerService(
            MODEL_PATH.as_posix()
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
        error_logger.error(traceback.format_exc())
        raise e

    frame_num = 0

    # ==========================================================================
    #                               LOOP PRINCIPAL
    # ==========================================================================
    empty_batches = 0
    saved_player_ids: List[int] = []
    try:
        for batch in video_stream:
            if not batch or len(batch) == 0:
                empty_batches += 1
                if empty_batches >= MAX_EMPTY_BATCHES:
                    debug_logger.debug("Múltiples batches vacíos consecutivos, finalizando procesamiento.")
                    break
                continue

            empty_batches = 0

            if time.time() - start_time > 1400: #MAX_PROCESSING_TIME * 4:
                debug_logger.debug("Tiempo de procesamiento excedido, finalizando.")
                break

            print(f"\n{'#'*60}\nProcesando batch de {len(batch)} frames...\n{'#'*60}\n")
            frame_num, updated_ids, updated_metrics = process_frame(
                match_id=match_id,
                video_batch=batch,
                frame_num=frame_num,
                db=db,
                tracker=tracker,
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
        heatmap_files = upload_heatmaps_for_extracted_players(db=db, match_id=match_id, extracted_player_ids=set(saved_player_ids))
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
        upload(
            key=f"{match_id}/reports/metrics_match_{match_id}.json",
            file_bytes=metrics_file.read_bytes(),
            file_type="application/json"
        )
        info_logger.info("Métricas escritas.")
        return heatmap_files
    except Exception as e:
        error_logger.error(f"Error processing video: {e}")
        error_logger.error(traceback.format_exc())
        raise e
