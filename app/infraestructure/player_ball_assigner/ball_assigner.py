from asyncio.log import logger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

import app.entities.utils.tools_context as context 


class BallAssigner:
    def __init__(
        self,
        max_distance_threshold: float = 2.5,
        angle_threshold: float = 45.0,
        cooldown_frames: int = 5,
        possession_threshold: float = 0.3,
    ):
        self.max_distance_threshold = max_distance_threshold
        self.angle_threshold = angle_threshold
        self.cooldown_frames = cooldown_frames
        self.possession_threshold = possession_threshold

        self.current_owner: Optional[int] = None
        self.owner_since_frame: int = -1
        self.possession_time: Dict[int, float] = {}
        self.last_ball_location: Optional[Tuple[float, float]] = None

    def update(
        self,
        ball_state: Optional[Dict[str, Any]],
        players: List[Dict[str, Any]],
        db: Session,
        dt: float,
        frame_number: int,
        scale: float = 1.0,
    ) -> Optional[int]:
        player_record = context.analysis_context.tools.player_records
        if ball_state is None:
            return self._release_owner(frame_number)

        bx, by = float(ball_state["x"]), float(ball_state["y"])
        d_max = self.max_distance_threshold * scale

        # Calcular velocidad del balón si es posible
        ball_velocity = 0.0
        if self.last_ball_location is not None:
            ball_velocity = (
                np.hypot(
                    bx - self.last_ball_location[0], by - self.last_ball_location[1]
                )
                / dt
                if dt > 0
                else 0.0
            )
        self.last_ball_location = (bx, by)

        candidates = []
        for player in players:
            px, py = player["x"], player["y"]
            if not px or not py:
                continue

            dist = np.hypot(px - bx, py - by)
            if dist > d_max:
                continue

            # Calcular ángulo jugador → balón
            vx, vy = player.get("vx", 0.0), player.get("vy", 0.0)
            if np.hypot(vx, vy) > 0.1:
                dir_player = np.arctan2(vy, vx)
                dir_ball = np.arctan2(by - py, bx - px)
                angle_diff = abs((dir_player - dir_ball + np.pi) % (2 * np.pi) - np.pi)
            else:
                angle_diff = 0.0

            # Puntuación combinada de distancia y ángulo
            score = dist + (angle_diff / self.angle_threshold) * d_max * 0.5
            if angle_diff <= self.angle_threshold:
                candidates.append((player["player_id"], dist, angle_diff, score))

        if not candidates:
            return self._release_owner(frame_number)

        # Elegir mejor candidato basado en puntuación
        best_id, _, _, _ = min(candidates, key=lambda x: x[3])

        # Verificar si debemos cambiar de dueño
        change_result = self._should_change_owner(
            best_id=best_id,
            players=players,
            ball_location=(bx, by),
            max_distance=d_max,
            frame_number=frame_number,
            ball_velocity=ball_velocity,
        )

        if change_result is not None:
            best_id = change_result

        if best_id != self.current_owner and best_id is not None:
            self._change_owner(best_id, frame_number)

        # Actualizar tiempo de posesión
        if self.current_owner is not None:
            self.possession_time[self.current_owner] = (
                self.possession_time.get(self.current_owner, 0.0) + dt
            )

        # Actualizar estados en base de datos
        for player in players:
            is_owner = player["player_id"] == self.current_owner
            ball_possession_time = self.possession_time.get(player["player_id"], 0.0)

            payload = {
                "has_ball": is_owner,
                "ball_owner_id": self.current_owner if is_owner else None,
                "ball_possession_time": ball_possession_time,
                "ball_x": bx,
                "ball_y": by,
            }

            player_record.patch_state(
                player_id=int(player["player_id"]),
                frame_index=frame_number,
                updates=payload,
            )

        return self.current_owner

    def _change_owner(self, new_id: int, frame: int) -> None:
        logger.debug(
            f"[BallAssigner] owner {self.current_owner} -- {new_id} at frame {frame}"
        )
        self.current_owner = new_id
        self.owner_since_frame = frame

    def _release_owner(self, frame: int) -> None:
        if self.current_owner is None:
            return
        logger.debug(
            f"[BallAssigner] owner {self.current_owner} released at frame {frame}"
        )
        self.current_owner = None
        self.owner_since_frame = -1

    def _should_change_owner(
        self,
        best_id: int,
        players: List[Dict[str, Any]],
        ball_location: Tuple[float, float],
        max_distance: float,
        frame_number: int,
        ball_velocity: float,
    ) -> Optional[int]:
        bx, by = ball_location

        if self.current_owner is None:
            return best_id

        if best_id == self.current_owner:
            return self.current_owner

        current_frames = frame_number - self.owner_since_frame
        if current_frames < self.cooldown_frames:
            return self.current_owner

        owner = next((p for p in players if p["player_id"] == self.current_owner), None)
        if not owner:
            return best_id

        dist_owner = np.hypot(owner["x"] - bx, owner["y"] - by)

        if ball_velocity > 5.0:  # Ajustar según necesidad
            return best_id

        if dist_owner <= max_distance * 1.2:
            return self.current_owner

        return best_id

    def get_possession_time(self, player_id: int) -> float:
        return self.possession_time.get(player_id, 0.0)

    def get_current_owner(self) -> Optional[int]:
        return self.current_owner
