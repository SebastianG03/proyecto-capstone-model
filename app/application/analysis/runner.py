import json
from pathlib import Path
import time
import tracemalloc

from tqdm import tqdm

from app.core.config import BATCH_SIZE, DEBUG, MAX_EMPTY_BATCHES
from app.entities.services.video_anotator import VideoAnotator
from app.infraestructure.plotting import generate_diagrams

from app.infraestructure.services.video_processing_service import check_video, get_total_frames
from app.infraestructure.trackers import TrackerService
import app.entities.utils.global_values_store as value_store

from app.application.analysis.process import process_frame
from app.logger.memory_tracker import MemoryReporter
from app.logger.process_time_reporter import ProcessTimeReporter
from app.shared.analysis_tools import AnalysisTools
from app.infraestructure.services import (
    upload_heatmaps_for_extracted_players,
    upload,
    R2Downloader,
    prepare_model,
    read_video,
)
import traceback

from app.utils.routes import ANOTATED_VIDEOS_DIR, BALL_MODEL_PATH, INPUT_VIDEOS_DIR, METRICS_DIR, PLAYER_MODEL_PATH
from app.logger import debug_logger, error_logger, info_logger


def run_analysis(video_name: str, match_id: int) -> dict[int, str] | None:
    metrics_file = METRICS_DIR / f"metrics_match_{match_id}.json"
    time_reporter = ProcessTimeReporter(logger=debug_logger, match_id=match_id)
    frame_num = 0
    empty_batches = 0
    tracemalloc.start()
    start_time = time.time()
    memory_reporter = MemoryReporter(
        match_id=match_id,
        alert_threshold_mb=3000
    )
    
    db = value_store.globals.connection_manager.create_session()
    value_store.globals.session = db

    metrics = {
        "processing_time": [],
        "memory_usage": [],
        "ball_detection": {"detected": 0, "interpolated": 0},
        "interpolation_error": 0.0,
        "velocity_inconsistencies": {"players": 0, "referees": 0},
    }

    prepare_model(model_path=BALL_MODEL_PATH, source_path=BALL_MODEL_PATH.parent)

    downloader = R2Downloader()

    # download_path = Path(INPUT_VIDEOS_DIR, video_name)
    download_path = Path(r"C:\Users\Usuario\Desktop\temp\res\Partido corto 2.mp4")
    # downloader.build_destination_path(
    #     key=video_name, base_dir=INPUT_VIDEOS_DIR.as_posix()
    # )
    # downloader.stream_download(
    #     key=video_name, destination_path=download_path.as_posix()
    # )
    print(f"Video descargado en {download_path.as_posix()}")

    batch_middle = BATCH_SIZE // 2
    batch_proportion = batch_middle // 2
    is_video = check_video(download_path.as_posix())
    total_frames = get_total_frames(download_path.as_posix())
    progress = tqdm(
        total=total_frames,
        desc="Procesando video",
        unit="frame(s)"
    )
    
    if not is_video:
        error_logger.error("Error: Video not found")
        raise FileNotFoundError
    
    video_stream = read_video(download_path.as_posix(), batch_size=BATCH_SIZE)
    
    value_store.globals.video_anotator = VideoAnotator(
        output_path=Path(f"{ANOTATED_VIDEOS_DIR}/{video_name[:10]}_{match_id}.mp4"),
        fps=value_store.globals.fps,
        frame_size=value_store.globals.frame_size,
        colors=value_store.globals.anotated_colors,
        window_name=f"Annotated Video of match {match_id}",
        show_preview=True
    )

    if not video_stream: 
        error_logger.error("Error: No frames read from video")
        raise Exception("El video no es valido, esta vacio o no se encuentra disponible en la nube.")

    try:
        tracker = TrackerService(
            ball_model_path=BALL_MODEL_PATH.as_posix(),
            player_model_path=PLAYER_MODEL_PATH.as_posix()
            )

        first_batch = next(video_stream)
        first_frame, _ = first_batch[0]
        first_batch = None

        if not first_frame.any():
            error_logger.error("Error: First frame is empty")
            return
        info_logger.info("Inicializando servicios de analisis...")
        tools = AnalysisTools()
        tools.start(first_frame=first_frame, match_id=match_id)
        first_frame = None
    except StopIteration as st:
        error_logger.error("Error: No frames read from video")
        raise st
    except Exception as e:
        error_logger.error(f"Error initializing services: {e}")
        error_logger.error(traceback.format_exc())
        raise e

    try:
        for batch in video_stream:
            if not batch or len(batch) == 0:
                empty_batches += 1
                if empty_batches >= MAX_EMPTY_BATCHES:
                    debug_logger.debug(
                        "Multiples batches vacios consecutivos, finalizando procesamiento."
                    )
                    break
                continue

            empty_batches = 0

            # if time.time() - start_time > 500:
            #     debug_logger.debug("Tiempo de procesamiento excedido, finalizando.")
            #     break
            
            if frame_num > 500 and DEBUG:
                break
            
            progress.update(len(batch))

            # frames = [
            #     batch[0],
            #     batch[batch_middle - batch_proportion],
            #     batch[batch_middle],
            #     batch[batch_middle + batch_proportion],
            #     batch[-1]
            # ]

            frame_num = process_frame(
                match_id=match_id,
                video_batch=batch,
                frame_num=frame_num,
                db=db,
                tracker=tracker,
                metrics=metrics,
                export_data_file=metrics_file,
                time_reporter=time_reporter,
            )
            time_reporter.publish()
            info_logger.info(
                f"Batch procesado. Frames hasta ahora: {frame_num + len(batch)}"
            )
            try:
                memory_reporter.after_loop(label=f"Frame_{frame_num}", local_vars=locals())
            except:
                
                break
        
        progress.close()
        info_logger.info("Procesamiento finalizado.")
        generate_diagrams(db)
        info_logger.info("Diagramas generados.")
        heatmap_files = upload_heatmaps_for_extracted_players(
            db=db, match_id=match_id
        )
        info_logger.info("Heatmaps subidos.")
        
        tools.analysis_data_collector.export_to_csv()
        info_logger.info("Datos de analisis exportados a CSV.")

        total_time = time.time() - start_time
        metrics.update({
            "Frames por minuto": (frame_num / total_time) * 60,
            "Frames procesados": frame_num,
            "Tiempo total de procesamiento": f"{total_time / 60:.2f} minutos",
        })

        info_logger.info("\n" + "=" * 50)
        info_logger.info("        RESUMEN FINAL DEL PROCESAMIENTO")
        info_logger.info("=" * 50)
        info_logger.info(f"Tiempo total: {total_time / 60:.2f} min")
        # print(f"Memoria maxima usada: {max(metrics['memory_usage']):.2f} MB")
        info_logger.info(
            f"Frames balon detectado: {metrics['ball_detection']['detected']}"
        )
        info_logger.info(
            f"Frames balon interpolado: {metrics['ball_detection']['interpolated']}"
        )
        info_logger.info(f"Escribiendo metricas a {metrics_file.as_posix()}...")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        upload(
            key=f"{match_id}/reports/metrics_match_{match_id}.json",
            file_bytes=metrics_file.read_bytes(),
            file_type="application/json",
        )
        info_logger.info("Metricas escritas.")
        return heatmap_files
    except Exception as e:
        error_logger.error(f"Error processing video: {e}")
        error_logger.error(traceback.format_exc())
        raise e
