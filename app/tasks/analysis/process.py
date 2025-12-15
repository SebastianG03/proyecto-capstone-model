import json
from pathlib import Path
import tracemalloc
from typing import List
from cv2.typing import MatLike
import numpy as np
from sqlalchemy.orm import Session

from app.modules.services.video_processing_service import extract_player_images
from app.modules.trackers.tracker_service import TrackerService
from app.tasks.analysis.assign_ball import assign_ball_to_player
from app.tasks.analysis.process_tracks import process_tracks_and_position
from app.modules.services.bbox_processor_service import calculate_area_boundary_ends
from app.tasks.analysis_tools import AnalysisTools
from app.utils.routes import OUTPUT_IMAGES_DIR
from app.logger import *

def process_frame(
    frame_num: int,
    video_batch: List[tuple[MatLike, float]],
    db: Session,
    tracker: TrackerService,
    tools: AnalysisTools,
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
    
    try:
        for frame, dt in video_batch:
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
                continue

            # -------------------------------------------------------
            # 2. TRACKING DE OBJETOS (jugadores + balón)
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 2: Procesando frame en el tracker...")
                process_tracks_and_position(
                    tools=tools,
                    frame=frame,
                    frame_num=frame_num,
                    db=db,
                    tracker=tracker,
                    camera_movement=camera_movement,
                    pixels_to_meters=pixels_to_meters,
                )
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error en process_tracks_and_position: {e}")
                continue

            # -------------------------------------------------------
            # 3. MÉTRICAS DE DETECCIÓN DEL BALÓN
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 3: Calculando métricas de detección del balón...")
                ball_frames = tools.ball_records.get_all()
                detected = sum(1 for t in ball_frames if t.get_bbox() is not None)
                total = len(ball_frames)
                metrics["ball_detection"] = {
                    "detected": detected,
                    "interpolated": total - detected,
                }
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error calculando métricas de balón: {e}")
                metrics["ball_detection"] = {"detected": 0, "interpolated": 0}

            # -------------------------------------------------------
            # 4. ESTIMAR VELOCIDAD / DISTANCIA
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 4: Estimando velocidad y distancia del último jugador...")
                last_player = tools.player_records.get_last(db)
                if last_player:
                    tools.speed_and_distance.process_track(
                        frame_num=frame_num,
                        track_id=int(f'{last_player.player_id}'),
                        track=last_player,
                        pixels_to_meters=pixels_to_meters,
                        camera_scale=tools.camera_movement_estimator.get_current_scale(),
                        db=db,
                    )
                else:
                    info_logger.info("[ProcessRun] No hay último jugador para estimar velocidad/distancia.")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error estimando velocidad/distancia: {e}")

            # -------------------------------------------------------
            # 5. ASIGNACIÓN DEL BALÓN A JUGADOR
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 5: Asignando balón a jugador...")
                players = tools.player_records.get_all()
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

            # -------------------------------------------------------
            # 6. ASIGNAR EQUIPO
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 6: Asignando equipo...")
                last_player = tools.player_records.get_last(db)
                if last_player:
                    tools.team_assigner.get_player_team(frame, last_player, db)
                else:
                    info_logger.info("[ProcessRun] No hay último jugador para asignar equipo.")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error asignando equipo: {e}")

            # -------------------------------------------------------
            # 7. EXTRAER IMÁGENES
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 7: Extrayendo imágenes de jugadores...")
                last_player = tools.player_records.get_last(db)
                if not last_player:
                    info_logger.info(f"[ProcessRun] No hay jugador para extraer imágenes, se continúa.")
                    continue

                updated_counts, updated_last, saved_id = extract_player_images(
                    frame=frame,
                    frame_index=frame_num,
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

            # -------------------------------------------------------
            # 8. EXPORTAR DATOS DEL FRAME
            # -------------------------------------------------------
            try:
                info_logger.info("[ProcessRun] Paso 8: Exportando datos del frame...")
                last_player = tools.player_records.get_last(db)
                ball_frame = tools.ball_records.get_last(db)
                snapshot = tracemalloc.take_snapshot()
                total_mem = sum(stat.size for stat in snapshot.statistics("lineno")) / (1024 * 1024)
                export_data = {
                    "frame_num": frame_num,
                    "frame_time": f"{dt:.4f} seconds",
                    "metrics": metrics.copy(),
                    "player_image_counts": player_image_counts.copy(),
                    "last_frame_taken": last_frame_taken.copy(),
                    "saved_player_ids": saved_player_ids.copy(),
                    "player_data": last_player.to_dict() if last_player else None,
                    "ball_data": ball_frame.to_dict() if ball_frame else None,
                    "memory_usage_mb": total_mem if 'total_mem' in locals() else None,
                }
                export_data_file.write_text(json.dumps(export_data, ensure_ascii=False) + "\n", encoding="utf-8")
            except Exception as e:
                error_logger.error(f"[Frame {frame_num}] Error escribiendo exportación: {e}")

            print(f"[Frame {frame_num}] procesado correctamente.")

        return frame_num, saved_player_ids, metrics

    except Exception as e:
        # Este catch solo se activa si algo sale fuera del loop de frames
        error_logger.error(f"Error fuera del loop de frames: {e}")
        raise e
    finally:
        # Asegurar flush de logs
        for hdlr in info_logger.handlers[:]:
            hdlr.flush()

# import json
# from pathlib import Path
# import tracemalloc
# from typing import List
# from cv2.typing import MatLike
# from sqlalchemy.orm import Session

# from app.entities.models.PlayerState import PlayerStateModel
# from app.modules.services.video_processing_service import extract_player_images
# from app.modules.trackers.tracker_service import TrackerService
# from app.tasks.analysis.assign_ball import assign_ball_to_player
# from app.tasks.analysis.process_tracks import process_tracks_and_position
# from app.tasks.analysis_tools import AnalysisTools
# from app.utils.routes import OUTPUT_IMAGES_DIR
# from app.logger import *

# def process_frame(
#     frame_num: int,
#     video_batch: List[tuple[MatLike, float]],
#     db: Session,
#     tracker: TrackerService,
#     tools: AnalysisTools,
#     metrics: dict,
#     images_per_player: int,
#     saved_player_ids: List[int],
#     export_data_file: Path
# ): 
#     player_image_counts: dict = {}
#     last_frame_taken: dict = {}
#     try:
#         for frame, dt in video_batch:
#             frame_num += 1
#             print(f"\n{'='*20} Procesando frame {frame_num} {'='*20}\n")
#             print(f"Tiempo desde último frame: {dt:.4f} segundos")
#             # -------------------------------------------------------
#             # 1. Estimar movimiento de cámara
#             # -------------------------------------------------------
#             print("Estimando movimiento de cámara...")
#             camera_movement = tools.camera_movement_estimator.update(frame)

#             # -------------------------------------------------------
#             # 2. TRACKING DE OBJETOS (jugadores + balón)
#             # -------------------------------------------------------
#             print("Procesando frame en el tracker...")
#             process_tracks_and_position(
#                 tools=tools,
#                 frame=frame,
#                 frame_num=frame_num,
#                 db=db,
#                 tracker=tracker,
#                 camera_movement=camera_movement,
#             )


#             # -------------------------------------------------------
#             # 4. MÉTRICAS DE DETECCIÓN DEL BALÓN
#             # -------------------------------------------------------
#             debug_logger.debug("Obteniendo métricas de detección del balón...")
#             ball_frames = tools.ball_records.get_all()
#             debug_logger.debug(f"Total frames con balón: {len(ball_frames)}")
#             if len(ball_frames) == 1:                
#                 metrics["ball_detection"] = {
#                     "detected": 1,
#                     "interpolated": 0,
#                 }
#             elif len(ball_frames) > 1:
#                 detected = sum(1 for t in ball_frames if t.get_bbox() is not None)
#                 print(f"Frames con balón detectado: {detected}")
#                 total = len(ball_frames)
#                 print(f"Total frames: {total}")

#                 metrics["ball_detection"] = {
#                     "detected": detected,
#                     "interpolated": total - detected,
#                 }


#             debug_logger.debug("Estimando velocidad y distancia del último jugador...")
#             last_player = tools.player_records.get_last(db)
#             debug_logger.debug("Ultimo jugador para obtener la velocidad y distancia: ", last_player is not None)
#             try:
#                 if last_player:
#                     debug_logger.debug(f"Último jugador para distancia: {last_player.player_id}")
#                     tools.speed_and_distance.process_track(
#                         frame_num=frame_num,
#                         track_id=int(f'{last_player.player_id}'),
#                         track=last_player,
#                         db=db,
#                     )
#                     debug_logger.debug("Velocidad y distancia estimadas.")
#                 else:
#                     debug_logger.debug("No hay último jugador para estimar velocidad/distancia.")
#             except Exception as e:
#                 error_logger.error(f"Error estimando velocidad/distancia para el jugador {last_player.player_id}: {e}")
#                 pass

#             # -------------------------------------------------------
#             # 6. ASIGNACIÓN DEL BALÓN A UN JUGADOR
#             # -------------------------------------------------------
#             players = tools.player_records.get_all()
#             if ball_frames is not None or len(ball_frames) > 0:
#                 assign_ball_to_player(
#                     ball_records=ball_frames,
#                     players=players,
#                     tools=tools,
#                     db=db,
#                     frame_index=frame_num,
#                     dt=dt
#                 )
            
#             print("Asignacion de equipo")
#             last_player = tools.player_records.get_last(db)
#             team = tools.team_assigner.get_player_team(frame, last_player, db)
#             print("Equipo asignado. Equipo: ", team)
            
#             print("Extrayendo imagenes de jugadores.")

#             last_player = tools.player_records.get_last(db)
#             print("Último jugador obtenido: ", last_player is not None)
#             if not last_player:
#                 print("No hay jugador para extraer imagenes, saltandom frame...")
#                 continue
#             updated_player_image_counts, updated_last_frame_taken, saved_player_id = extract_player_images(
#             frame=frame,
#             frame_index=frame_num,
#             player=last_player,
#             images_per_player=images_per_player,
#             output_folder=OUTPUT_IMAGES_DIR.as_posix(),
#             player_image_counts=player_image_counts,
#             last_frame_taken=last_frame_taken,
#             )
#             player_image_counts.update(updated_player_image_counts)
#             last_frame_taken.update(updated_last_frame_taken)
#             if saved_player_id is not None:
#                 saved_player_ids.append(saved_player_id)
#             print("Imágenes extraídas.")
            
#             if all(count >= images_per_player for count in player_image_counts.values()):
#                 last_frame_taken.clear() 
            
#             print(f"Frame procesado. {frame_num}")

#             print("Datos de exportación del frame:")
#             export_data = {
#                 "frame_num": frame_num,
#                 "frame_time": f'{dt:.4f} seconds',
#                 "metrics": metrics,
#                 "player_image_counts": player_image_counts,
#                 "last_frame_taken": last_frame_taken,
#                 "saved_player_ids": saved_player_ids,
#                 "player_data": last_player.to_dict() if last_player else None,
#                 "ball_data": ball_frames[-1].to_dict() if ball_frames else None
#             }
#             print(export_data)
#             info_logger.info(f"Frame {frame_num} data: {export_data}")
            
#             print("Escribiendo datos de exportación al archivo...")
#             chars = export_data_file.open("a", encoding="utf-8").write(json.dumps(export_data) + "\n")
#             print("Datos escritos al archivo. Escrito: ", chars, " caracteres.")

#             snapshot = tracemalloc.take_snapshot()
#             total_mem = sum(stat.size for stat in snapshot.statistics("lineno")) / (1024 * 1024)

#             if not metrics["memory_usage"]:
#                 metrics["memory_usage"].append(total_mem)
#         return frame_num, saved_player_ids, metrics
#     except Exception as e:
#         print(f"Error procesando frame {frame_num}: {e}")
#         raise e
#     finally:
#         for hdlr in info_logger.handlers[:]:
#             hdlr.flush()
#             hdlr.close()
#             info_logger.removeHandler(hdlr)
            
