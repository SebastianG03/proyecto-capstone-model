import json
from multiprocessing import process
from pathlib import Path
import time
import tracemalloc
import logging
from logging.handlers import RotatingFileHandler
from typing import List
from xml.sax import handler

from app.entities.collections import TrackCollectionBall, TrackCollectionPlayer 
from app.entities.models import BallEventModel, PlayerStateModel
from app.modules.camera_movement_estimator import \
    CameraMovementEstimator
from app.modules.player_ball_assigner import \
    PlayerBallAssigner
from app.modules.plotting import generate_diagrams
from app.modules.services import read_video, extract_player_images
from app.modules.services.r2_download import R2Downloader
from app.modules.services.verify_model import prepare_model
from app.modules.services.video_processing_service import extract_player_images
from app.modules.speed_and_distance_estimator import SpeedAndDistanceEstimator
from app.modules.team_assigner import TeamAssigner
from app.entities.trackers import ( 
    BallTracker)
from app.modules.trackers import TrackerService
from app.modules.view_transformer import ViewTransformer
from sqlalchemy.orm import Session

from app.tasks.upload_heatmaps import upload_heatmaps_for_extracted_players
from app.utils.routes import INPUT_VIDEOS_DIR, MODELS_DIR, OUTPUT_IMAGES_DIR, OUTPUT_REPORTS_DIR
from cv2.typing import MatLike

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("info.log", maxBytes=1000000, backupCount=5)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

async def run_analysis(db: Session, video_name: str, match_id: int) -> None:
    try:
        export_data_file = OUTPUT_REPORTS_DIR / f"export_data_match_{match_id}.txt"
        export_data_file.parent.mkdir(parents=True, exist_ok=True)
        print("Archivo de reporteria creado: ", export_data_file.exists())
        
        if not export_data_file.exists():
            export_data_file.touch()
            print("Volviendo a intentar: Archivo de reporteria creado en: ", export_data_file.as_posix())
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
        print("Error: No frames read from video")
        return

    try:
        tracker = TrackerService(
            model_path.as_posix()
        )

        # -----------------------------
        # COLECCIONES
        # -----------------------------
        player_records = TrackCollectionPlayer(db)
        player_records.orm_model = PlayerStateModel

        ball_records = TrackCollectionBall(db)
        ball_records.orm_model = BallEventModel

        view_transformer = ViewTransformer()
        speed_and_distance = SpeedAndDistanceEstimator()
        team_assigner = TeamAssigner()
        player_assigner = PlayerBallAssigner()
    except Exception as e:
        print(f"Error initializing services: {e}")
        raise e

     # -----------------------------
     # FRAME INICIAL PARA CAMERA MOVEMENT
     # -----------------------------
    try:
        first_batch = next(video_stream)
        first_frame, _ = first_batch[0]
    except StopIteration:
        print("Error: Video is empty")
        return

    if not first_frame.any():
        print("Error: First frame is empty")
        return
    camera_movement_estimator = CameraMovementEstimator(first_frame)
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
                print("Múltiples batches vacíos consecutivos, finalizando procesamiento.")
                break
            continue

        empty_batches = 0

        if time.time() - start_time > max_processing_time:
            print("Tiempo de procesamiento excedido, finalizando.")
            break

        print(f"\n{'#'*60}\nProcesando batch de {len(batch)} frames...\n{'#'*60}\n")
        frame_num, updated_ids, updated_metrics = process_frame(
            video_batch=batch,
            frame_num=frame_num,
            db=db,
            tracker=tracker,
            player_records=player_records,
            ball_records=ball_records,
            camera_movement_estimator=camera_movement_estimator,
            view_transformer=view_transformer,
            speed_and_distance=speed_and_distance,
            team_assigner=team_assigner,
            player_assigner=player_assigner,
            metrics=metrics,
            images_per_player=images_per_player,
            saved_player_ids=saved_player_ids,
            export_data_file=export_data_file
        )
        saved_player_ids.extend(updated_ids)
        metrics.update(updated_metrics)
        print(f"Batch procesado. Frames hasta ahora: {frame_num + len(batch)}")

    print(f"Jugadores con imágenes extraídas {saved_player_ids}")

    generate_diagrams(db)
    print("Diagramas generados.")
    await upload_heatmaps_for_extracted_players(db=db, match_id=match_id, extracted_player_ids=set(saved_player_ids))
    print("Heatmaps subidos.")

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("        RESUMEN FINAL DEL PROCESAMIENTO")
    print("=" * 50)
    print(f"Tiempo total: {total_time/60:.2f} min")
    # print(f"Memoria máxima usada: {max(metrics['memory_usage']):.2f} MB")
    print(f"Frames balón detectado: {metrics['ball_detection']['detected']}")
    print(f"Frames balón interpolado: {metrics['ball_detection']['interpolated']}")


def process_frame(
    frame_num: int,
    video_batch: List[tuple[MatLike, float]],
    db: Session,
    tracker: TrackerService,
    player_records: TrackCollectionPlayer,
    ball_records: TrackCollectionBall,
    camera_movement_estimator: CameraMovementEstimator,
    view_transformer: ViewTransformer,
    speed_and_distance: SpeedAndDistanceEstimator,
    team_assigner: TeamAssigner,
    player_assigner: PlayerBallAssigner,
    metrics: dict,
    images_per_player: int,
    saved_player_ids: List[int],
    export_data_file: Path
): 
    player_image_counts: dict = {}
    last_frame_taken: dict = {}
    try:
        for frame, dt in video_batch:
            frame_num += 1
            print(f"\n{'='*20} Procesando frame {frame_num} {'='*20}\n")
            print(f"Tiempo desde último frame: {dt:.4f} segundos")
            # -------------------------------------------------------
            # 1. Estimar movimiento de cámara
            # -------------------------------------------------------
            print("Estimando movimiento de cámara...")
            camera_movement = camera_movement_estimator.update(frame)

            # -------------------------------------------------------
            # 2. TRACKING DE OBJETOS (jugadores + balón)
            # -------------------------------------------------------
            print("Procesando frame en el tracker...")
            process_tracks_and_position(
                player_records=player_records,
                ball_records=ball_records,
                frame=frame,
                frame_num=frame_num,
                db=db,
                tracker=tracker,
                camera_movement_estimator=camera_movement_estimator,
                camera_movement=camera_movement,
                view_transformer=view_transformer
            )
                 
            
            # -------------------------------------------------------
            # 3. VALIDAR QUE EL TRACKER DE BALÓN ES CORRECTO
            # -------------------------------------------------------
            print("Obteniendo tracker de balón...")
            ball_tracker = tracker.get_tracker("ball")
            print("Tracker de balón obtenido.")
            if not isinstance(ball_tracker, BallTracker):
                raise TypeError("Retrieved tracker is not an instance of BallTracker")

            # -------------------------------------------------------
            # 4. MÉTRICAS DE DETECCIÓN DEL BALÓN
            # -------------------------------------------------------
            print("Obteniendo métricas de detección del balón...")
            ball_frames = ball_records.get_all()
            print(f"Total frames con balón: {len(ball_frames)}")
            if len(ball_frames) == 1:                
                metrics["ball_detection"] = {
                    "detected": 1,
                    "interpolated": 0,
                }
            elif len(ball_frames) > 1:
                detected = sum(1 for t in ball_frames if t.get_bbox() is not None)
                print(f"Frames con balón detectado: {detected}")
                total = len(ball_frames)
                print(f"Total frames: {total}")

                metrics["ball_detection"] = {
                    "detected": detected,
                    "interpolated": total - detected,
                }

            # -------------------------------------------------------
            # 5. ESTIMAR VELOCIDAD/DISTANCIA DEL ÚLTIMO JUGADOR
            # -------------------------------------------------------
            print("Estimando velocidad y distancia del último jugador...")
            last_player = player_records.get_last(db)
            print("Último jugador obtenido: ", last_player is not None)
            if last_player:
                print(f"Último jugador: {last_player.player_id}")
                speed_and_distance.process_track(
                    frame_num=frame_num,
                    track_id=int(f'{last_player.player_id}'),
                    track=last_player,
                    db=db,
                    model_class=PlayerStateModel,
                )
                print("Velocidad y distancia estimadas.")

            # -------------------------------------------------------
            # 6. ASIGNACIÓN DEL BALÓN A UN JUGADOR
            # -------------------------------------------------------
            players = player_records.get_all()
            team_ball_control = []
            
            if ball_frames is not None or len(ball_frames) > 0:
                team_ball_control = assign_ball_to_player(
                    ball_records=ball_frames,
                    players=players,
                    team_assigner=team_assigner,
                    player_assigner=player_assigner,
                    team_ball_control=team_ball_control,
                    db=db,
                    frame_index=frame_num,
                    frame=frame,
                )

            print("Extrayendo imagenes de jugadores.")

            last_player = player_records.get_last(db)
            print("Último jugador obtenido: ", last_player is not None)
            if not last_player:
                print("No hay jugador para extraer imagenes, saltandom frame...")
                continue
            updated_player_image_counts, updated_last_frame_taken, saved_player_id = extract_player_images(
            frame=frame,
            frame_index=frame_num,
            player=last_player,
            images_per_player=images_per_player,
            output_folder=OUTPUT_IMAGES_DIR.as_posix(),
            player_image_counts=player_image_counts,
            last_frame_taken=last_frame_taken,
            )
            player_image_counts.update(updated_player_image_counts)
            last_frame_taken.update(updated_last_frame_taken)
            if saved_player_id is not None:
                saved_player_ids.append(saved_player_id)
            print("Imágenes extraídas.")
            
            if all(count >= images_per_player for count in player_image_counts.values()):
                last_frame_taken.clear() 
            
            print(f"Frame procesado. {frame_num}")

            print("Datos de exportación del frame:")
            export_data = {
                "frame_num": frame_num,
                "frame_time": f'{dt:.4f} seconds',
                "metrics": metrics,
                "player_image_counts": player_image_counts,
                "last_frame_taken": last_frame_taken,
                "saved_player_ids": saved_player_ids,
                "player_data": last_player.to_dict() if last_player else None,
                "ball_data": ball_frames[-1].to_dict() if ball_frames else None
            }
            print(export_data)
            logger.info(f"Frame {frame_num} data: {export_data}")
            
            print("Escribiendo datos de exportación al archivo...")
            chars = export_data_file.open("a", encoding="utf-8").write(json.dumps(export_data) + "\n")
            print("Datos escritos al archivo. Escrito: ", chars, " caracteres.")

            snapshot = tracemalloc.take_snapshot()
            total_mem = sum(stat.size for stat in snapshot.statistics("lineno")) / (1024 * 1024)

            if not metrics["memory_usage"]:
                metrics["memory_usage"].append(total_mem)
        return frame_num, saved_player_ids, metrics
    except Exception as e:
        print(f"Error procesando frame {frame_num}: {e}")
        raise e
    finally:
        for hdlr in logger.handlers[:]:
            hdlr.flush()
            hdlr.close()
            logger.removeHandler(hdlr)
            
def process_tracks_and_position(
    player_records: TrackCollectionPlayer,
    ball_records: TrackCollectionBall,
    frame: MatLike,
    frame_num: int,
    db: Session,
    tracker: TrackerService,
    camera_movement_estimator: CameraMovementEstimator,
    camera_movement: tuple[float, float],
    view_transformer: ViewTransformer
):
    print("Procesando tracks y posiciones...")
    try:
        for collection in (player_records, ball_records):
            print("Obteniendo tracks de objetos...")
            scale = camera_movement_estimator.get_current_scale()
            tracker.get_object_tracks(frame, frame_num, db)

            print("Actualizando último track...")
            last_track = collection.get_last(db)
            if last_track is None:
                print("No hay track para actualizar, saltando...")
                continue
            print(f"Último track ID: {last_track.to_dict().get('player_id', None)}")

            # Calcular centro y bbox inmediatamente
            print("Añadiendo posición al track...")
            tracker.add_position_to_track(db, last_track)
            print(f"Posición añadida: ({last_track.x}, {last_track.y})")

            # Aplicar compensación de movimiento de cámara
            print("Ajustando posiciones según movimiento de cámara...")
            camera_movement_estimator.add_adjust_positions_to_tracks(
                db=db,
                camera_movement_per_frame=camera_movement,
                scale=scale,
                track=last_track
            )
            print("Posiciones ajustadas.")

            # Homografía al campo 2D
            print("Aplicando transformación de vista...")
            view_transformer.add_transformed_positions(db)
    except Exception as e:
        print(f"Error procesando tracks y posiciones: {e}")
        raise e
    print("Procesamiento de tracks y posiciones completado.")

def assign_ball_to_player(
    ball_records: List[BallEventModel],
    players: List[PlayerStateModel],
    team_assigner: TeamAssigner,
    player_assigner: PlayerBallAssigner,
    team_ball_control: List[int],
    frame_index: int,
    db: Session,
    frame: MatLike
):
    try:
        print("Asignando balón a jugador...")
        player_records = TrackCollectionPlayer(db)
        print("Colección de jugadores obtenida. Longitud actual: ", len(players))
        for ball_track in ball_records:
            # FRAME SIN BALÓN
            if not ball_track:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                print("No hay balón en este frame, asignación por defecto.")
                continue

            # Solo un balón por frame
            print("Obteniendo detalle del balón...")
            ball_bbox = ball_track.get_bbox()
            print(f"Bbox del balón: {ball_bbox}, numero de elementos en bbox: {len(ball_bbox) if ball_bbox else 'N/A'}")
            print("Bbox data type: ", type(ball_bbox))
            if ball_bbox is None:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                print("No hay balón en este frame, asignación por defecto.")
                continue

            # -----------------------------
            # ASIGNACIÓN A JUGADOR
            # -----------------------------
            print("Asignando balón a jugador...")
            assigned_player_id = player_assigner.assign_ball_to_player(
                players=players,
                ball_bbox=ball_bbox
            )

            if assigned_player_id == -1:
                print("No hay jugador asignado, asignación por defecto.")
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # Ubicar el jugador asignado en ese frame real
            print(f"Jugador asignado: {assigned_player_id}")
            player = player_records.get_record_for_frame(
                assigned_player_id,
                frame_index
            )
            print(f"Jugador encontrado en frame: {player is not None}")

            if not player:
                print("Jugador no encontrado, asignación por defecto.")
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # Marcar posesión
            print("Marcando posesión del jugador...")
            player_records.patch(
                player.to_dict()["id"],
                {"has_ball": True}
            )

            # Obtener equipo
            print("Obteniendo equipo del jugador...")
            team = team_assigner.get_player_team(frame, player)
            print(f"Equipo del jugador: {team}")
            team_ball_control.append(team if team is not None else -1)
            print("Posesión marcada.")
        print("Asignación de balón completada.")
        return team_ball_control
    except Exception as e:
        print(f"Error en asignación de balón: {e}")
        raise e
