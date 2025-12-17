
from asyncio.log import logger
from typing import Any, Dict, List, Union
from sqlalchemy.orm import Session

import numpy as np
from app.entities.collections.track_collections import TrackCollectionPlayer
from app.entities.models.BallState import BallEventModel
from app.modules.player_ball_assigner.ball_assigner import BallAssigner
from app.modules.services.bbox_processor_service import (
    get_center_of_bbox, measure_scalar_distance)
from app.entities.models.PlayerState import PlayerStateModel

class PlayerBallAssigner():
    def __init__(
        self,
        maximum_player_ball_distance: int = 70,
        angle_threshold: float = 45.0,
        cooldown_frames: int = 5,
        fps: float = 30):
        self.maximum_player_ball_distance = maximum_player_ball_distance
        self.angle_threshold = angle_threshold
        self.cooldown_frames = cooldown_frames
        self.fps = fps
        self.last_frame: int = -1

        self.ball_assigner = BallAssigner(
            max_distance_threshold=maximum_player_ball_distance,
            angle_threshold=angle_threshold,
            cooldown_frames=cooldown_frames,
            possession_threshold=0.3
        )

    def assign_ball_to_player(
        self,
        players: List[PlayerStateModel],
        ball_event: BallEventModel,
        db: Session,
        dt: float = 1,
        scale: float = 1.0,
        frame_number: int = 0
        ) -> int:
        try:
            print("[PlayerBallAssigner] assign_ball_to_player llamado en frame ", frame_number)
            player_record = TrackCollectionPlayer(db)
            # 1. Crear ball_state (compatibilidad con Kalman)
            ball_bbox = ball_event.get_bbox() if ball_event else None
            if not ball_bbox or len(ball_bbox) < 4:
                print("[PlayerBallAssigner] No hay bbox de balón válido.")
                return -1

            print(f"[PlayerBallAssigner] ball_bbox: {ball_bbox}")

            # 3. Convertir PlayerStateModel → dict ligero
            players_dict = [
                {
                    "id": int(f'{p.id}'),
                    "player_id": int(f'{p.player_id}'),
                    "x": float(f'{p.x}'),
                    "y": float(f'{p.y}'),
                    "ball_possession_time": float(f'{p.ball_possession_time}') if p.ball_possession_time is not None else 0.0,
                    "vx": float(f'{p.speed}') * np.cos(np.deg2rad(45)),
                    "vy": float(f'{p.speed}') * np.sin(np.deg2rad(45))
                }
                for p in players if p.x is not None and p.y is not None and p.speed is not None
            ]
            ball_state = {
                "x": float(f'{ball_event.x}'),
                "y": float(f'{ball_event.y}'),
                "vx": float(f'{ball_event.vx}'),
                "vy": float(f'{ball_event.vy}')
            }

            # 4. Asignación con lógica nueva
            print(f"[PlayerBallAssigner] players_dict: {players_dict}")
            print("Actualizando instancia de BallAssigner...")
            owner_id = self.ball_assigner.update(
                ball_state=ball_state,
                players=players_dict,
                db=db,
                dt=dt,
                scale=scale,
                frame_number=frame_number
            )
            print(f"[PlayerBallAssigner] owner_id determinado: {owner_id}")

            for player in players:
                print(f"[PlayerBallAssigner] player: {player}")
                is_owner = (int(f'{player.player_id}') == owner_id)
                payload = {
                    "has_ball": is_owner,
                    "ball_owner_id": owner_id if is_owner else None
                }
                if is_owner:
                    print("[PlayerBallAssigner] Jugador es dueño del balón.")
                    # acumular tiempo (fps puede venir de config)
                    payload["ball_possession_time"] = (float(f'{player.ball_possession_time}') or 0.0) + dt
            # 6. Devolver mismo tipo que antes
                print(f"[PlayerBallAssigner] payload: {payload}")
                player_record.patch(int(f'{player.id}'), payload)
                print(f"[PlayerBallAssigner] Player {player.player_id} updated: {payload}")
            return owner_id if owner_id is not None else -1

        except Exception as e:
            logger.exception("Error en PlayerBallAssigner: ", exc_info=e)
            print(f"Error en PlayerBallAssigner: {e}")
            return -1

    # def assign_ball_to_player(
    #         self,
    #         players: List[PlayerStateModel],
    #         ball_bbox):
    #     try:
    #         ball_position = get_center_of_bbox(ball_bbox)

    #         min_distance = float('inf')
    #         closest_player_id = -1
            
    #         for player in players:
    #             bbox = player.get_bbox()

    #             if not bbox or len(bbox) < 4:
    #                 continue

    #             left_foot  = (bbox[0], bbox[3])
    #             right_foot = (bbox[2], bbox[3])

    #             dist_left  = measure_scalar_distance(left_foot, ball_position)
    #             dist_right = measure_scalar_distance(right_foot, ball_position)
    #             distance   = min(dist_left, dist_right)

    #             if distance < self.maximum_player_ball_distance and distance < min_distance:
    #                 min_distance = distance
    #                 closest_player_id = player.to_dict()['id']

    #         return closest_player_id
    #     except Exception as e:
    #         print(f"Error asignando balón a jugador: {e}")
    #         raise e
