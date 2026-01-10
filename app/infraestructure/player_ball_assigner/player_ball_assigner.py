from asyncio.log import logger
from typing import List
from sqlalchemy.orm import Session

import numpy as np
from app.entities.collections import TrackCollectionPlayer
from app.entities.models.BallState import BallEventModel
from app.infraestructure.player_ball_assigner.ball_assigner import BallAssigner
from app.entities.models.PlayerModels import PlayerState, Player

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
        players: List[PlayerState],
        ball_event: BallEventModel,
        db: Session,
        dt: float = 1,
        scale: float = 1.0,
        frame_number: int = 0
        ) -> int:
        try:
            if not ball_event or ball_event.x is None or ball_event.y is None:
                return -1

            # Calcular velocidades reales basadas en posiciones consecutivas
            players_dict = []
            for p in players:
                if p.x is not None and p.y is not None:
                    # Calcular velocidad real si hay datos históricos
                    vx, vy = 0.0, 0.0
                    if hasattr(p, 'prev_x') and hasattr(p, 'prev_y') and p.prev_x is not None and p.prev_y is not None:
                        vx = (p.x - p.prev_x) / dt if dt > 0 else 0.0
                        vy = (p.y - p.prev_y) / dt if dt > 0 else 0.0
                    elif p.speed is not None and float(f'{p.speed}') > 0:
                        # Usar dirección anterior si existe
                        if hasattr(p, 'prev_direction') and p.prev_direction is not None:
                            vx = p.speed * np.cos(p.prev_direction)
                            vy = p.speed * np.sin(p.prev_direction)
                        else:
                            vx = p.speed * 0.5  # Dirección por defecto
                            vy = p.speed * 0.5
                    
                    players_dict.append({
                        "id": int(f'{p.id}'),
                        "player_id": int(f'{p.player_id}'),
                        "x": float(f'{p.x}'),
                        "y": float(f'{p.y}'),
                        "ball_x": float(f'{ball_event.x}'),
                        "ball_y": float(f'{ball_event.y}'),
                        "ball_possession_time": float(f'{p.ball_possession_time}') if p.ball_possession_time is not None else 0.0,
                        "vx": vx,
                        "vy": vy
                    })

            ball_state = {
                "x": float(f'{ball_event.x}'),
                "y": float(f'{ball_event.y}'),
                "vx": float(f'{ball_event.vx}') if ball_event.vx is not None else 0.0,
                "vy": float(f'{ball_event.vy}') if ball_event.vy is not None else 0.0
            }

            owner_id = self.ball_assigner.update(
                ball_state=ball_state,
                players=players_dict,
                db=db,
                dt=dt,
                scale=scale,
                frame_number=frame_number
            )

            # Actualizar estados de jugadores
            for player in players:
                is_owner = (int(f'{player.player_id}') == owner_id)
                possession_time = self.ball_assigner.get_possession_time(int(f'{player.player_id}'))
                
                payload = {
                    "has_ball": is_owner,
                    "ball_owner_id": owner_id if is_owner else None,
                    "ball_x": float(f'{ball_event.x}'),
                    "ball_y": float(f'{ball_event.y}'),
                    "ball_possession_time": possession_time
                }
                player_record = TrackCollectionPlayer(db)
                player_record.patch(int(f'{player.id}'), payload)

            return owner_id if owner_id is not None else -1

        except Exception as e:
            logger.exception("Error en PlayerBallAssigner: ", exc_info=e)
            return -1
