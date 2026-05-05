from asyncio.log import logger
from typing import List
from sqlalchemy.orm import Session

import numpy as np
from app.entities.collections import TrackCollectionPlayer
from app.entities.models.BallState import BallEventModel
import app.entities.utils.tools_context as context 
from app.infraestructure.player_ball_assigner.ball_assigner import BallAssigner
from app.entities.models.PlayerModels import PlayerState
from app.logger.logger import info_logger

class PlayerBallAssigner:
    def __init__(
        self,
        maximum_player_ball_distance: int = 70,
        angle_threshold: float = 45.0,
        cooldown_frames: int = 5,
        fps: float = 30,
    ):
        self.maximum_player_ball_distance = maximum_player_ball_distance
        self.angle_threshold = angle_threshold
        self.cooldown_frames = cooldown_frames
        self.fps = fps
        self.last_frame: int = -1

        self.ball_assigner = BallAssigner(
            max_distance_threshold=maximum_player_ball_distance,
            angle_threshold=angle_threshold,
            cooldown_frames=cooldown_frames,
            possession_threshold=0.3,
        )

    def assign_ball_to_player(
        self,
        players: List[PlayerState],
        ball_event: BallEventModel,
        db: Session,
        dt: float = 1,
        scale: float = 1.0,
        frame_number: int = 0,
    ) -> int:
        try:
            if not ball_event or ball_event.x is None or ball_event.y is None:
                return -1

            players_dict = []
            for p in players:
                vx, vy = 0.0, 0.0
                if p.x is not None and p.y is not None:
                    prev_player_state = context.analysis_context.tools.player_records.get_previous_state(int(f'{p.player_id}'), frame_number)
                    xf = (float(f"{p.x}") * scale, float(f"{p.y}") * scale)

                    if prev_player_state is not None and prev_player_state.x is not None and prev_player_state.y is not None:
                        xo = (float(f"{prev_player_state.x}") * scale, float(f"{prev_player_state.y}") * scale)

                        delta_x = np.array([xo[0] - xf[0], xo[1] - xf[1]])
                        vo = delta_x / dt
                        acceleration = (2 * (delta_x - vo * dt)) / dt ** 2
                        vf = vo + acceleration * dt
                        vx, vy = vf[0], vf[1]

                    players_dict.append({
                        "id": int(f"{p.id}"),
                        "player_id": int(f"{p.player_id}"),
                        "x": float(f"{p.x}"),
                        "y": float(f"{p.y}"),
                        "ball_x": float(f"{ball_event.x}"),
                        "ball_y": float(f"{ball_event.y}"),
                        "ball_possession_time": float(f"{p.ball_possession_time}")
                        if p.ball_possession_time is not None
                        else 0.0,
                        "vx": vx,
                        "vy": vy,
                    })

            ball_state = {
                "x": float(f"{ball_event.x}"),
                "y": float(f"{ball_event.y}"),
                "vx": float(f"{ball_event.vx}") if ball_event.vx is not None else 0.0,
                "vy": float(f"{ball_event.vy}") if ball_event.vy is not None else 0.0,
            }
            
            info_logger.info(f"[PlayerBallAssigner] Jugadores en frame: {players_dict}")

            owner_id = self.ball_assigner.update(
                ball_state=ball_state,
                players=players_dict,
                db=db,
                dt=dt,
                scale=scale,
                frame_number=frame_number,
            )

            for player in players:
                is_owner = int(f"{player.player_id}") == owner_id
                possession_time = self.ball_assigner.get_possession_time(
                    int(f"{player.player_id}")
                )

                payload = {
                    "has_ball": is_owner,
                    "ball_owner_id": owner_id if is_owner else None,
                    "ball_x": float(f"{ball_event.x}"),
                    "ball_y": float(f"{ball_event.y}"),
                    "ball_possession_time": possession_time,
                }
                player_record = TrackCollectionPlayer()
                player_record.patch(int(f"{player.player_id}"), payload)

            return owner_id if owner_id is not None else -1

        except Exception as e:
            logger.exception("Error en PlayerBallAssigner: ", exc_info=e)
            return -1
