from asyncio.log import logger
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.entities.collections.track_collections import TrackCollectionPlayer
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerState import PlayerStateModel


class BallAssigner:
    def __init__(
        self,
        max_distance_threshold: float = 2.5,
        angle_threshold: float = 45.0,
        cooldown_frames: int = 5,
        possession_threshold: float = 0.3
        ):
        """
        Initializes a BallAssigner object.

        Parameters
        ----------
        max_distance_threshold : float, optional
            The maximum distance (in meters) between a player and the ball
            for which the ball is assigned to the player. Defaults to 2.5.
        angle_threshold : float, optional
            The minimum angle (in degrees) between the player's direction and the
            direction from the player to the ball for which the ball is assigned to
            the player. Defaults to 45.0.
        cooldown_frames : int, optional
            The number of frames for which the ball is not assigned to any player after
            a player has been assigned the ball. Defaults to 5.
        possession_threshold : float, optional
            The minimum time (in seconds) for which a player must be in possession of
            the ball for which the ball is assigned to the player. Defaults to 0.3.
        """
        self.max_distance_threshold = max_distance_threshold
        self.angle_threshold = angle_threshold
        self.cooldown_frames = cooldown_frames
        self.possession_threshold = possession_threshold

        self.current_owner: Optional[int] = None
        self.owner_since_frame: int = -1
        self.possession_time: Dict[int, float] = {}

    def update(
        self,
        ball_state: Optional[Dict[str, Any]],
        players: List[Dict[str, Any]],
        db: Session,
        dt: float,
        frame_number: int,
        scale: float = 1.0,
    ) -> Optional[int]:
        """
        Updates the current owner of the ball.

        Parameters
        ----------
        ball_state : Optional[Dict[str, Any]]
            The current state of the ball.
        players : List[Dict[str, Any]]
            List of players currently in the scene.
        db : Session
            The current database session.
        dt : float
            The time difference between the current frame and the last frame.
        frame_number : int
            The current frame number.
        scale : float, optional
            The scale of the camera. Defaults to 1.0.

        Returns
        -------
        Optional[int]
            The ID of the player who is currently in possession of the ball, or None if no player is in possession.
        """
        player_record = TrackCollectionPlayer(db)
        if ball_state is None:
            return self._release_owner(frame_number)

        bx, by = float(ball_state["x"]), float(ball_state["y"])
        d_max = self.max_distance_threshold * scale

        candidates = []
        for player in players:
            px, py = player["x"], player["y"]
            print("[BallAssigner] Player", player["player_id"], "at", (px, py))
            if not px or not py:
                continue
            dist = np.hypot(px - bx, py - by)
            if dist > d_max:
                continue

            # ángulo jugador → balón
            vx, vy = player.get("vx", 0.0), player.get("vy", 0.0)
            if np.hypot(vx, vy) > 0.1:  # solo si tiene velocidad
                dir_player = np.arctan2(vy, vx)
                dir_ball = np.arctan2(by - py, bx - px)
                angle_diff = abs((dir_player - dir_ball + np.pi) % (2 * np.pi) - np.pi)
            else:
                angle_diff = 0.0  # sin info de dirección, no penalizar

            if angle_diff <= self.angle_threshold:
                candidates.append((player["player_id"], dist, angle_diff))


        if not candidates:
            return self._release_owner(frame_number)

        best_id, _, _ = min(candidates, key=lambda x: x[1])

        self._update_current_owner(
            best_id=best_id,
            ball_location=(bx, by),
            candidates=candidates,
            players=players,
            max_distance=d_max,
            frame_number=frame_number
        )

        # ------------------------------------------------------------------
        # 5. Cambio oficial de posesión
        # ------------------------------------------------------------------
        if best_id != self.current_owner:
            self._change_owner(best_id, frame_number)

        # ------------------------------------------------------------------
        # 6. Actualizar modelos IN-PLACE
        # ------------------------------------------------------------------
        for player in players:
            is_owner = (player["player_id"] == self.current_owner)
            payload = {
                "has_ball": is_owner,
                "ball_owner_id": self.current_owner if is_owner else None
            }
            if is_owner:
                payload.update({"ball_possession_time": (float(f'{player["ball_possession_time"]}') or 0.0) + dt})
            player_record.patch(player["id"], payload)

        return self.current_owner

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _change_owner(self, new_id: int, frame: int) -> None:
        """
        Changes the current owner of the ball.

        Parameters
        ----------
        new_id : int
            The new owner ID.
        frame : int
            The current frame number.

        Returns
        -------
        None
        """
        logger.debug(f"[BallAssigner] owner {self.current_owner} → {new_id} at frame {frame}")
        self.current_owner = new_id
        self.owner_since_frame = frame

    def _release_owner(self, frame: int) -> None:
        """
        Releases the current owner of the ball.

        Parameters
        ----------
        frame : int
            The current frame number.

        Returns
        -------
        None
        """
    
        if self.current_owner is None:
            return
        logger.debug(f"[BallAssigner] owner {self.current_owner} released at frame {frame}")
        self.current_owner = None
        self.owner_since_frame = -1

    def _update_current_owner(
        self,
        candidates: List[Tuple],
        best_id: int,
        players: List[Dict[str, Any]],
        ball_location: Tuple[float, float],
        max_distance: float,
        frame_number: int) -> Optional[int]:
        """
        Decides whether to update the current owner of the ball or not.

        Parameters
        ----------
        candidates : List[Tuple]
            List of tuples containing the distance of each player from the ball.
        best_id : int
            Player id of the closest player to the ball.
        players : List[PlayerStateModel]
            List of PlayerStateModel objects.
        ball_location : Tuple[float, float]
            Tuple containing the x and y coordinates of the ball.
        max_distance : float
            Maximum distance from the ball at which ownership can be transferred.
        frame_number : int
            Current frame number.

        Returns
        -------
        Optional[int]
            Player id of the new owner of the ball, or None if no change was made.
        """
        bx, by = ball_location

        if self.current_owner is None or best_id == self.current_owner:
            return None

        current_frames = frame_number - self.owner_since_frame
        if self.owner_since_frame != -1 and current_frames > self.cooldown_frames:
            return None

        owner = next((p for p in players if p["player_id"] == self.current_owner), None)

        if not owner: return None
        
        dist_owner = np.hypot(owner["x"] - bx, owner["y"] - by)
        if dist_owner <= max_distance * 1.5:
            return self.current_owner
        return None

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def get_possession_time(self, player_id: int) -> float:
        """
        Devuelve el tiempo de posesión del balón para un jugador.

        Parameters
        ----------
        player_id : int
            Identificador del jugador.

        Returns
        -------
        float
            Tiempo de posesión del balón para ese usuario (en segundos), o 0.0 si no se encuentra.
        """

        return self.possession_time.get(player_id, 0.0)

    def get_current_owner(self) -> Optional[int]:
        """
        Devuelve el jugador actual que posee el balón (o None si no hay)
        """
        return self.current_owner