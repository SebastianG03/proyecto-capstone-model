
from typing import Any, Dict, List

import numpy as np
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
        fps: int = 30):
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
        ball_bbox: List[float]
        ) -> int:
        try:
            # 1. Crear ball_state (compatibilidad con Kalman)
            if not ball_bbox or len(ball_bbox) < 4:
                return -1
            cx, cy = get_center_of_bbox(ball_bbox)
            ball_state: Dict[str, Any] = {
                "x": cx,
                "y": cy,
                "vx": 0.0,
                "vy": 0.0
            }

            # 2. Escala de cámara (si existe)
            #    Como no tenemos acceso al frame aquí, usamos scale=1.0
            #    Si más adelante se pasa el frame, se puede inyectar.
            scale = 1.0  # TODO: inyectar CameraMovementEstimator si se tiene

            # 3. Convertir PlayerStateModel → dict ligero
            players_dict = [
                {
                    "player_id": p.player_id,
                    "x": p.x,
                    "y": p.y,
                    "vx": p.speed * np.cos(np.deg2rad(45)),  # aprox
                    "vy": p.speed * np.sin(np.deg2rad(45))
                }
                for p in players
            ]

            # 4. Asignación con lógica nueva
            frame = players[0].frame_index if players else 0
            owner_id = self.ball_assigner.update(
                ball_state=ball_state,
                players=players_dict,
                db=None,
            )

            # 5. Actualizar campos de posesión en los objetos SQLAlchemy
            for p in players:
                is_owner = (p.player_id == owner_id)
                p.has_ball = is_owner
                p.ball_owner_id = owner_id if is_owner else None
                if is_owner:
                    # acumular tiempo (fps puede venir de config)
                    dt = 1.0 / self.fps
                    p.ball_possession_time = (p.ball_possession_time or 0.0) + dt

            # 6. Devolver mismo tipo que antes
            return owner_id if owner_id is not None else -1

        except Exception as e:
            logger.exception("Error en PlayerBallAssignerV2")
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
