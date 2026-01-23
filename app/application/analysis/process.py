from collections import defaultdict
import json
from os import devnull
from pathlib import Path
import time
import tracemalloc
from typing import List, Optional
from cv2.typing import MatLike
import numpy as np
from sqlalchemy.orm import Session

import supervision as sv

from app.entities.models.PlayerModels import Player, PlayerState
from app.entities.utils.global_values_store import GlobalValuesStore
from app.infraestructure.services.video_processing_service import extract_player_images
from app.infraestructure.trackers.goal_scorer_detector import GoalScorerDetector
from app.infraestructure.trackers.goal_tracker import GoalTracker
from app.infraestructure.trackers.tracker_service import TrackerService
from app.application.analysis.assign_ball import assign_ball_to_player
from app.application.analysis.process_tracks import process_tracks_and_position
from app.infraestructure.services.bbox_processor_service import calculate_area_boundary_ends
from app.entities.utils import analysis_context
from app.utils.routes import MODEL_GOALS_PATH, OUTPUT_IMAGES_DIR
from app.logger import *

def process_frame(
    match_id: int,
    frame_num: int,
    video_batch: List[tuple[MatLike, float]],
    db: Session,
    tracker: TrackerService,
    metrics: dict,
    images_per_player: int, 
    saved_player_ids: List[int],
    export_data_file: Path
) -> tuple[int, List[int], dict]:
    """
    Procesa un lote de frames de video aislándolos entre sí:
    si un frame falla se ignora y se continúa con el siguiente.
    """
    player_image_counts: dict = {}
    last_frame_taken: dict = {}
    start_time = time.time()
    globals = GlobalValuesStore()
    tools = analysis_context.tools
    numbers_data = defaultdict(lambda: defaultdict(float))
    info_logger.info(f"[ProcessRun] Iniciando procesamiento de lote de frames en timestamp {start_time}")
    errors = 0
    goal_yolo = GoalTracker(MODEL_GOALS_PATH)
    goal_scorer = GoalScorerDetector(iou_threshold=0.0, pixel_threshold=200.0)
    try:
        for frame, _ in video_batch:
            from app.infraestructure.services.database import create_temporary_database
            db = create_temporary_database(match_id)[0]
            actual_time = time.time()
            dt = actual_time - start_time
            globals.update(timestamp=dt)
            frame_num += 1
            pixels_to_meters = 0.1048
            area_boundarys = calculate_area_boundary_ends(frame)
            if area_boundarys is not None:
                d_px = np.linalg.norm(area_boundarys[0] - area_boundarys[1])
                pixels_to_meters = float(11 / d_px)
            debug_logger.debug(f"[ProcessRun] Factor pixels_to_meters calculado: {pixels_to_meters}")

            print(f"\n{'='*20} Procesando frame {frame_num} {'='*20}\n")
            print(f"Tiempo desde último frame: {dt:.4f} segundos")

            # -------------------------------------------------------
            # 1. Estimar movimiento de cámara
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 1: Estimando movimiento de cámara...")
                camera_movement = tools.camera_movement_estimator.update(frame)
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error estimando movimiento de cámara: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error
                continue

            # -------------------------------------------------------
            # 2. TRACKING DE OBJETOS (jugadores + balón)
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 2: Procesando frame en el tracker...")
                process_tracks_and_position(
                    frame=frame,
                    frame_num=frame_num,
                    db=db,
                    tracker=tracker,
                    camera_movement=camera_movement,
                    pixels_to_meters=pixels_to_meters,
                )
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error en process_tracks_and_position: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error
                continue


            # -------------------------------------------------------
            # 4. ESTIMAR VELOCIDAD / DISTANCIA
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 4: Estimando velocidad y distancia del último jugador...")
                last_state = tools.player_records.get_last()
                if last_state:
                    tools.speed_and_distance.process_track(
                        frame_num=frame_num,
                        track_id=int(f'{last_state.player_id}'),
                        track=last_state,
                        pixels_to_meters=pixels_to_meters,
                        camera_scale=tools.camera_movement_estimator.get_current_scale(),
                        db=db,
                        dt=dt,
                    )
                else:
                    info_logger.info("[ProcessRun] No hay último jugador para estimar velocidad/distancia.")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error estimando velocidad/distancia: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error
                continue

            # -------------------------------------------------------
            # 5. ASIGNACIÓN DEL BALÓN A JUGADOR, DEPENDE DE LA EJECUCION DEL PUNTO 4
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 5: Asignando balón a jugador...")
                players = tools.player_records.get_all_states()
                ball_frames = tools.ball_records.get_all()
                if ball_frames:
                    assign_ball_to_player(
                        ball_records=ball_frames,
                        players=players,
                        tools=tools,
                        db=db,
                        frame_index=frame_num,
                        dt=dt
                    )
                else:
                    info_logger.info("[ProcessRun] No hay frames de balón para asignar.")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error asignando balón: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error
                continue

            # -------------------------------------------------------
            # 6. ASIGNAR EQUIPO, ACCION INDEPENDIENTE
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 6: Asignando equipo...")
                state = db.query(PlayerState).order_by(PlayerState.id.desc()).first()
                
                if state is None:
                    info_logger.info("[ProcessRun] No hay estado de jugador para asignar equipo.")
                    continue

                last_state = tools.player_records.get_last()
                if last_state:
                    tools.team_assigner.assign_team_colors(frame=frame, players=tools.player_records.get_all_states())
                    tools.team_assigner.get_player_team(
                        frame=frame,
                        frame_num=frame_num,
                        record=last_state,
                        db=db)
                else:
                    info_logger.info("[ProcessRun] No hay último jugador para asignar equipo.")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error asignando equipo: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error
                continue

            try:
                last_state = tools.player_records.get_last()
                if last_state is not None and last_state.get_bbox() is not None:
                    player_id = int(f'{last_state.player_id}')
                    crop = tools.number_recognizer._crop_dorsal_region(frame, last_state.get_bbox())
                    if crop.size == 0:
                        info_logger.info("[ProcessRun] Crop del dorsal vacío, omitiendo reconocimiento.")
                    else:
                        proc = tools.number_recognizer._preprocess(crop)
                        info_logger.info("[ProcessRun] Reconociendo número de jugador...")

                        def update_best_num(num: Optional[int], conf: float):
                            if num is not None and conf > 0.60:
                                debug_logger.debug(f"[ProcessRun] Número {num} con confianza {conf:.4f} añadido al jugador {player_id}.")
                                numbers_data[player_id][num] += conf

                        must_flush = tools.trocr_buffer.push(proc, update_best_num)
                        if must_flush:
                            info_logger.info("[ProcessRun] Flushing buffer...")
                            tools.number_recognizer.flush_buffer(tools.trocr_buffer)

                    # decisión inmediata (con lo que haya hasta ahora)
                    best_num = max(numbers_data[player_id], key=numbers_data[player_id].get, default=None) # type: ignore
                    info_logger.info(f"[ProcessRun] Número de jugador reconocido: {best_num}")
                    debug_logger.debug(f"[ProcessRun] Datos completos de números: { {k: dict(v) for k, v in numbers_data.items()} }")
                    
                    player = tools.player_records.get_player(player_id)
                    if best_num is not None and player is not None:
                        info_logger.info(f"[ProcessRun] Actualizando número de jugador {best_num}...")
                        id = int(f'{player.id}')
                        tools.player_records.patch(id, {"shirt_number": best_num})
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error reconociendo número de jugador: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error
                continue
            
            try:
                info_logger.info("[ProcessRun] Detectando goles...")
                goal_detections = goal_yolo.predict(frame)
                info_logger.info(f"[ProcessRun] Detectados {len(goal_detections)} instancias de arcos.")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error en goal_yolo: {e}")
                goal_detections = sv.Detections.empty()
            
            try:
                info_logger.info("[ProcessRun] Actualizando goles...")
                scored, scorer_id = goal_scorer.update(
                    detections=goal_detections,
                    match_id=match_id,
                    frame_num=frame_num,
                    db=db
                )
                info_logger.info("[ProcessRun] Goles actualizados.")
                if scored and scorer_id is not None:
                    info_logger.info(f"[ProcessRun] ¡GOL de jugador {scorer_id}!")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error en goal_scorer: {e}")

            # -------------------------------------------------------
            # 7. EXTRAER IMÁGENES
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 7: Extrayendo imágenes de jugadores...")
                last_state = tools.player_records.get_last()
                last_player = tools.player_records.get_player(int(f'{last_state.player_id}')) if last_state else None
                if not last_state and not last_player:
                    info_logger.info(f"[ProcessRun] No hay jugador para extraer imágenes, se continúa.")
                    continue
                
                
                updated_counts, updated_last, saved_id = extract_player_images(
                    frame=frame,
                    frame_index=frame_num,
                    player_state=last_state,
                    player=last_player,
                    images_per_player=images_per_player,
                    output_folder=OUTPUT_IMAGES_DIR.as_posix(),
                    player_image_counts=player_image_counts,
                    last_frame_taken=last_frame_taken,
                )
                info_logger.info(f"[Frame {frame_num}] Imágenes de jugador extraídas correctamente.")
                debug_logger.debug(f"[ProcessRun] Objetos devueltos por extract_player_images: counts={updated_counts}, last_frame_taken={updated_last}, saved_id={saved_id}")
                player_image_counts.update(updated_counts)
                last_frame_taken.update(updated_last)
                if saved_id is not None:
                    saved_player_ids.append(saved_id)

                if all(count >= images_per_player for count in player_image_counts.values()):
                    debug_logger.debug(f"[ProcessRun] Se han alcanzado las imágenes requeridas para todos los jugadores. Limpiando last_frame_taken.")
                    last_frame_taken.clear()
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error extrayendo imágenes: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error
            # -------------------------------------------------------
            # 8. EXPORTAR DATOS DEL FRAME, accion independiente
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 8: Exportando datos del frame...")
                last_state = tools.player_records.get_last()
                ball_frame = tools.ball_records.get_last()
                snapshot = tracemalloc.take_snapshot()
                total_mem = sum(stat.size for stat in snapshot.statistics("lineno")) / (1024 * 1024)
                export_data = {
                    "frame_num": frame_num,
                    "frame_time": f"{dt:.4f} seconds",
                    "metrics": metrics.copy(),
                    "player_image_counts": player_image_counts.copy(),
                    "last_frame_taken": last_frame_taken.copy(),
                    "saved_player_ids": saved_player_ids.copy(),
                    "player_data": last_state.to_dict() if last_state else None,
                    "ball_data": ball_frame.to_dict() if ball_frame else None,
                    "memory_usage_mb": total_mem if 'total_mem' in locals() else None,
                }
                export_data_file.write_text(json.dumps(export_data, ensure_ascii=False) + "\n", encoding="utf-8")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error escribiendo exportación: {e}")
                error = update_error(errors)
                if error == -1: break
                errors = error

            print(f"[Frame {frame_num}] procesado correctamente.")

        return frame_num, saved_player_ids, metrics

    except Exception as e:
        error_logger.error(f"Error fuera del loop de frames: {e}")
        raise e
    finally:
        for hdlr in info_logger.handlers[:]:
            hdlr.flush()

def update_error(error: int):
    if error > 10:
        return -1
    return error + 1