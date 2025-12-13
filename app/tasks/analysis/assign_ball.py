from typing import List
from sqlalchemy.orm import Session

from app.entities.collections.track_collections import TrackCollectionPlayer
from app.entities.models import PlayerStateModel, BallEventModel
from app.tasks.analysis_tools import AnalysisTools


def assign_ball_to_player(
    ball_records: List[BallEventModel],
    players: List[PlayerStateModel],
    tools: AnalysisTools,
    dt: float,
    frame_index: int,
    db: Session,
):
    try:
        print("Iniciando asignación de balón a jugador...")
        player_records = TrackCollectionPlayer(db)
        print("Colección de jugadores obtenida. Longitud actual: ", len(players))
        print("Iterando sobre registros de balón, registros actuales en colección: ", len(ball_records))
        for ball_track in ball_records:
            # FRAME SIN BALÓN
            if not ball_track:
                print("No hay balón en este frame, asignación por defecto.")
                continue

            # Solo un balón por frame
            print("Obteniendo detalle del balón...")
            ball_bbox = ball_track.get_bbox()
            print(f"Bbox del balón: {ball_bbox}, numero de elementos en bbox: {len(ball_bbox) if ball_bbox else 'N/A'}")
            print("Bbox data type: ", type(ball_bbox))
            if ball_bbox is None:
                print("No hay balón en este frame, asignación por defecto.")
                continue

            # -----------------------------
            # ASIGNACIÓN A JUGADOR
            # -----------------------------
            print("Asignando balón a jugador...")
            assigned_player_id = tools.player_ball_assigner.assign_ball_to_player(
                players=players,
                ball_event=ball_track,
                db=db,
                dt=dt,
                frame_number=frame_index,
                scale=tools.camera_movement_estimator.get_current_scale()
            )
            print(f"Jugador asignado ID (track_id): {assigned_player_id}")

            if assigned_player_id == -1:
                print("No hay jugador asignado, asignación por defecto.")
                continue

            # Ubicar el jugador asignado en ese frame real
            print(f"Jugador asignado: {assigned_player_id}")
            # player = player_records.get_record_for_frame(
            #     assigned_player_id,
            #     frame_index
            # )
            # print(f"Jugador encontrado en frame: {player is not None}")

            # if not player:
            #     print("Jugador no encontrado, asignación por defecto.")
            #     continue

            # # Marcar posesión
            # print("Marcando posesión del jugador...")
            # player_records.patch(
            #     player.to_dict()["id"],
            #     {"has_ball": True}
            # )

            # # Obtener equipo
            # print("Obteniendo equipo del jugador...")
            # team = team_assigner.get_player_team(frame, player)
            # print(f"Equipo del jugador: {team}")
            print("Posesión marcada.")
        print("Asignación de balón completada.")
    except Exception as e:
        print(f"Error en asignación de balón: {e}")
        raise e
