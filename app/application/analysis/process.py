from collections import defaultdict
import json
from pathlib import Path
import time
import tracemalloc
from typing import List, Optional, Tuple
from cv2.typing import MatLike
import numpy as np
from sqlalchemy.orm import Session

from app.entities.models.PlayerModels import PlayerState
import app.entities.utils.global_values_store as value_store
from app.infraestructure.services.video_processing_service import extract_player_images
from app.infraestructure.trackers.goal_scorer_detector import GoalScorerDetector
from app.infraestructure.trackers.goal_tracker import GoalTracker
from app.infraestructure.trackers.tracker_service import TrackerService
from app.application.analysis.assign_ball import assign_ball_to_player
from app.application.analysis.process_tracks import process_tracks_and_position
from app.infraestructure.services.bbox_processor_service import (
    calculate_area_boundary_ends,
)
from app.entities.utils import analysis_context
from app.logger.process_time_reporter import ProcessTimeReporter
from app.utils.routes import MODEL_GOALS_PATH, OUTPUT_IMAGES_DIR
from app.logger import info_logger, error_logger, debug_logger



def process_frame(
    match_id: int,
    frame_num: int,
    video_batch: List[Tuple[MatLike, float]],
    db: Session,
    tracker: TrackerService,
    metrics: dict,
    export_data_file: Path,
    time_reporter: ProcessTimeReporter,
) -> int:
    """
    Procesa un lote de frames de video aislándolos entre sí:
    si un frame falla se ignora y se continúa con el siguiente.
    """
    start_time = time.time()
    tools = analysis_context.tools
    numbers_data = defaultdict(lambda: defaultdict(float))
    info_logger.info(
        f"[ProcessRun] Iniciando procesamiento de lote de frames en timestamp {start_time}"
    )
    goal_yolo = GoalTracker(MODEL_GOALS_PATH)
    goal_scorer = GoalScorerDetector(iou_threshold=0.0, pixel_threshold=200.0)
    
    time_reporter.start("get object tracks")
    time_reporter.stop("get object tracks")
    try:
        for frame, dt in video_batch:
            value_store.globals.update(timestamp=dt)
            frame_num += 1
            pixels_to_meters = 0.1048
            area_boundarys = calculate_area_boundary_ends(frame)
            if area_boundarys is not None:
                d_px = np.linalg.norm(area_boundarys[0] - area_boundarys[1])
                pixels_to_meters = float(11 / d_px)
            debug_logger.debug(
                f"[ProcessRun] Factor pixels_to_meters calculado: {pixels_to_meters}"
            )

            info_logger.info(f"\n{'=' * 20} Procesando frame {frame_num} {'=' * 20}\n")
            info_logger.info(f"Tiempo desde último frame: {dt:.4f} segundos")

            # -------------------------------------------------------
            # 1. Estimar movimiento de cámara
            # -------------------------------------------------------
            try:
                info_logger.info(
                    "[ProcessRun] Paso 1: Estimando movimiento de cámara..."
                )
                time_reporter.start("camera_movement_estimator")
                camera_movement = tools.camera_movement_estimator.update(frame)
                time_reporter.stop("camera_movement_estimator")
            except Exception as e:
                error_logger.error(
                    f"[Frame {frame_num}] Error estimando movimiento de cámara: {e}"
                )
                raise e

            # -------------------------------------------------------
            # 2. TRACKING DE OBJETOS (jugadores + balón)
            # -------------------------------------------------------

            tracker.get_object_tracks(frame, frame_num, db)

            try:
                info_logger.info(
                    "[ProcessRun] Paso 2: Procesando frame en el tracker..."
                )
                time_reporter.start("position updater (process tracks and position)")
                process_tracks_and_position(
                    db=db,
                    tracker=tracker,
                    camera_movement=camera_movement,
                    pixels_to_meters=pixels_to_meters,
                )
                time_reporter.stop("position updater (process tracks and position)")
            except Exception as e:
                error_logger.error(
                    f"[Frame {frame_num}] Error en process_tracks_and_position: {e}"
                )
                raise e
            
            try:
                last_players = tools.player_records.get_states_by_frame(frame_num)
                depths = []
                time_reporter.start("depth_estimator")
                if last_players is not None or len(last_players) > 0:
                    for player in last_players:
                        bbox = player.get_bbox()                
                        if bbox: 
                            depth = tools.depth_estimator.process_player_depth(
                            frame=frame, 
                            bbox=bbox, 
                            frame_num=frame_num, 
                            current_camera_scale=tools.camera_movement_estimator.get_current_scale()
                            )
                            if depth is not None:
                                depths.append(depth)
                
                if depths:
                    median_depth = np.median(depths)
                    value_store.globals.depth = median_depth
                    info_logger.info(f"[DepthEstimator] Profundidad estimada para el frame {frame_num}: {median_depth:.2f} metros, basado en {len(depths)} jugadores con profundidad estimada.")
                else:
                    info_logger.info(f"[DepthEstimator] No se pudieron estimar profundidades para el frame {frame_num}, se mantiene el anterior valor.")

                time_reporter.stop("depth_estimator")
            except Exception as e:
                error_logger.error(
                    f"[Frame {frame_num}] Error en process_tracks_and_position: {e}"
                )
                raise e

            # -------------------------------------------------------
            # 4. ESTIMAR VELOCIDAD / DISTANCIA
            # -------------------------------------------------------
            info_logger.info(
                "[ProcessRun] Paso 4: Estimando velocidad y distancia del último jugador..."
            )
            constant = tools.camera_movement_estimator.get_current_scale() * value_store.globals.depth * pixels_to_meters
            time_reporter.start("speed_and_distance")
            tools.speed_and_distance.process_track(
                frame_num=frame_num,
                constant=constant,
                db=db,
                dt=dt,
            )
            time_reporter.stop("speed_and_distance")

            # -------------------------------------------------------
            # 5. ASIGNACIÓN DEL BALÓN A JUGADOR, DEPENDE DE LA EJECUCION DEL PUNTO 4
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 5: Asignando balón a jugador...")
                players = tools.player_records.get_all_states()[:50]
                ball_frames = tools.ball_records.get_all()[:15]
                if ball_frames:
                    time_reporter.start("assign_ball_to_player")
                    assign_ball_to_player(
                        ball_records=ball_frames,
                        players=players,
                        tools=tools,
                        db=db,
                        frame_index=frame_num,
                        depth=value_store.globals.depth or 1.0,
                        pixels_to_meters=pixels_to_meters,
                        dt=dt,
                        logger=info_logger,
                    )
                    time_reporter.stop("assign_ball_to_player")
                else:
                    info_logger.info(
                        "[ProcessRun] No hay frames de balón para asignar."
                    )
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error asignando balón: {e}")
                raise e

            # -------------------------------------------------------
            # 6. ASIGNAR EQUIPO, ACCION INDEPENDIENTE
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 6: Asignando equipo...")
                state = db.query(PlayerState).order_by(PlayerState.id.desc()).first()

                if state is not None:
                    last_state = tools.player_records.get_last()
                    if last_state:
                        time_reporter.start("team assigner")
                        tools.team_assigner.assign_team_colors(
                            frame=frame, players=tools.player_records.get_all_states()
                        )
                        tools.team_assigner.get_player_team(
                            frame=frame, frame_num=frame_num, record=last_state, db=db
                        )
                        time_reporter.stop("team assigner")
                    else:
                        info_logger.info(
                            "[ProcessRun] No hay último jugador para asignar equipo."
                        )
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error asignando equipo: {e}")
                raise e

            try:
                info_logger.info("[ProcessRun] Iniciando el reconocimiento del numero del jugador")
                players = tools.player_records.get_states_by_frame(frame_num)
                info_logger.info("[ProcessRun] Ultimo estado del jugador obtenido")
                
                time_reporter.start("number recognizer")
                if players is not None and len(players) > 0:
                    players = [player.to_dict() for player in players]
                    for player in players:
                        player_id = int(player["player_id"])
                        crop = tools.number_recognizer._crop_dorsal_region(
                            frame, player["bbox"]
                        )
                        if crop.size == 0:
                            info_logger.info(
                                "[ProcessRun] Crop del dorsal vacío, omitiendo reconocimiento."
                            )
                        else:
                            proc = tools.number_recognizer._preprocess(crop)
                            info_logger.info(
                                "[ProcessRun] Reconociendo número de jugador..."
                            )

                            def update_best_num(num: Optional[int], conf: float):
                                if num is not None and conf > 0.35:
                                    debug_logger.debug(
                                        f"[ProcessRun] Número {num} con confianza "
                                        f"{conf:.4f} añadido al jugador {player_id}."
                                    )
                                    numbers_data[player_id][num] += conf
                            
                            if proc is not None:
                                must_flush = tools.trocr_buffer.push(proc, update_best_num)
                                if must_flush:
                                    info_logger.info("[ProcessRun] Flushing buffer...")
                                    tools.number_recognizer.flush_buffer(tools.trocr_buffer)
                        
                        player_numbers = numbers_data.get(player_id, {})
                        if player_numbers:
                            player_number = max(player_numbers, key=player_numbers.get) # type: ignore
                            if player_number is not None:
                                info_logger.info(
                                    f"[ProcessRun] Número de jugador reconocido: {player_number}"
                                )
                                info_logger.info(f"[ProcessRun] Paso 7: Reconociendo número de jugador: {player_number}")
                                player_db_id = tools.player_records.get_player_id(player_id)
                                if player_db_id and player_db_id != -1:
                                    tools.player_records.patch(
                                        player_db_id,
                                        {
                                            "shirt_number": player_number
                                        }
                                    )
                                    tools.analysis_data_collector.update_row(
                                        frame=frame_num,
                                        track_id=player_id,
                                        shirt_number=player_number)
                                    info_logger.info(f"[ProcessRun] Número de jugador actualizado exitosamente: {player_number}")
                                else:
                                    error_logger.error(f"[Frame {frame_num}] No se encontró Player con player_id {player_id}")
                    time_reporter.stop("number recognizer")
                else:
                    info_logger.info("[ProcessRun] No hay último jugador para reconocer número.")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error reconociendo número de jugador: {e}")
                raise e


            try:
                time_reporter.start("goal scorer")
                detections  = goal_yolo.predict(frame)
                frame_anotated = goal_yolo.annotate(frame, detections)
                # Modify the model path and versions, also update to only detect goals
                goal_detected, scorer_player_id = goal_scorer.updateScorer(
                    db=db,
                    detections=detections,
                    match_id=match_id,
                    frame_num=frame_num
                )
                time_reporter.stop("goal scorer")
                
                info_logger.info(
                    f"[Frame {frame_num}] Se detectaron {goal_detected} goles del jugador {scorer_player_id}."
                )
            except:
                raise Exception("Error al reconocer el Arco y detectar goles.") 
            
            for detected_object in value_store.globals.detected_object:
                detected_object_name = detected_object.name
                detected_object_id = detected_object.id
            
                if detected_object_name == "player":
                    state = analysis_context.tools.player_records.get_state_by_id(detected_object_id)
                    if state is not None:
                        player = analysis_context.tools.player_records.get_player(int(f"{state.player_id}"))
                        bbox = state.get_bbox()
                        info_logger.info(f"[PlayerDB] Bounding box: {bbox}, de tipo {type(bbox)}")
                        if bbox is not None and len(bbox) > 0:
                            speed = float(f"{state.speed}")
                            distance = float(f"{state.distance}")
                            number = f"{player.shirt_number if player is not None else ""}"
                            track_id = int(f"{state.player_id}")
                            frame_anotated = value_store.globals.video_anotator.annotate(frame_anotated, bbox, "player", f"id {track_id}, {number}, {speed:.2f} km/h, {distance:.2f} m, conf {detected_object.confidence:.2f}")
                if detected_object_name == "ball":
                    state = analysis_context.tools.ball_records.get_by_id(detected_object_id)
                    if state is not None:
                        bbox = state.get_bbox()
                        if bbox is not None and len(bbox) > 0:
                            frame_anotated = value_store.globals.video_anotator.annotate(frame_anotated, bbox, "ball", f"ball, conf: {detected_object.confidence:.2f}")

            value_store.globals.video_anotator.write_and_show(frame_anotated, frame_num)
            value_store.globals.reset_detected_object()


            # -------------------------------------------------------
            # 8. EXPORTAR DATOS DEL FRAME, accion independiente
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 8: Exportando datos del frame...")
                last_state = tools.player_records.get_last()
                ball_frame = tools.ball_records.get_last()
                snapshot = tracemalloc.take_snapshot()
                total_mem = sum(stat.size for stat in snapshot.statistics("lineno")) / (
                    1024 * 1024
                )
                export_data = {
                    "frame_num": frame_num,
                    "frame_time": f"{dt:.4f} seconds",
                    "metrics": metrics.copy(),
                    "player_data": last_state.to_dict() if last_state else None,
                    "ball_data": ball_frame.to_dict() if ball_frame else None,
                    "memory_usage_mb": total_mem if "total_mem" in locals() else None,
                }
                export_data_file.write_text(
                    json.dumps(export_data, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            except Exception as e:
                error_logger.error(
                    f"[Frame {frame_num}] Error escribiendo exportación: {e}"
                )
                raise e

            info_logger.info(f"[Frame {frame_num}] procesado correctamente.")

        return frame_num

    except Exception as e:
        error_logger.error(f"Error fuera del loop de frames: {e}")
        raise e
    finally:
        for hdlr in info_logger.handlers[:]:
            hdlr.flush()
        db.close()


def update_error(error: int):
    if error > 10:
        return -1
    return error + 1