from cv2.typing import MatLike
from sqlalchemy.orm import Session

from app.modules.trackers.tracker_service import TrackerService
from app.tasks.analysis_tools import AnalysisTools


def process_tracks_and_position(
    frame: MatLike,
    frame_num: int,
    db: Session,
    tracker: TrackerService,
    tools: AnalysisTools,
    camera_movement: tuple[float, float]
):
    print("Procesando tracks y posiciones...")
    try:
        for collection in (tools.player_records, tools.ball_records):
            print("Obteniendo tracks de objetos...")
            scale = tools.camera_movement_estimator.get_current_scale()
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
            tools.camera_movement_estimator.add_adjust_positions_to_tracks(
                db=db,
                camera_movement_per_frame=camera_movement,
                scale=scale,
                track=last_track
            )
            print("Posiciones ajustadas.")

            # Homografía al campo 2D
            print("Aplicando transformación de vista...")
            tools.view_transformer.add_transformed_positions(db)
    except Exception as e:
        print(f"Error procesando tracks y posiciones: {e}")
        raise e
    print("Procesamiento de tracks y posiciones completado.")
