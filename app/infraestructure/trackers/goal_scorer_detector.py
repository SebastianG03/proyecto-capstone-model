from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
from sqlalchemy.orm import Session

from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState
from app.logger import debug_logger, error_logger, info_logger


@dataclass
class BallPossessionSnapshot:
    frame: int
    player_id: int
    distance: float


class GoalScorerDetector:
    """
    Detecta GOL y asigna el tanto al jugador que más cerca estuvo
    del balón en los últimos <HISTORY_WINDOW> frames previos.
    """

    HISTORY_WINDOW = 30

    def __init__(
        self,
        iou_threshold: float = 0.0,
        pixel_threshold: float = 200.0,
        max_assign_distance: float = 170.0,
    ):
        self.iou_threshold = iou_threshold
        self.pixel_threshold = pixel_threshold
        self.max_assign_distance = max_assign_distance

        self._possession_history: deque[BallPossessionSnapshot] = deque(
            maxlen=self.HISTORY_WINDOW
        )

    # ------------------------------------------------------------------ #
    # Helpers geométricos
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bbox_center(bbox: list[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _distance_bbox(bbox_a: list[float], bbox_b: list[float]) -> float:
        return np.linalg.norm(
            np.array(GoalScorerDetector._bbox_center(bbox_a))
            - np.array(GoalScorerDetector._bbox_center(bbox_b))
        )

    @staticmethod
    def _ball_inside_goal(ball_bbox: list[float], goal_bbox: list[float]) -> bool:
        bx1, by1, bx2, by2 = ball_bbox
        gx1, gy1, gx2, gy2 = goal_bbox
        return bx1 >= gx1 and bx2 <= gx2 and by1 >= gy1 and by2 <= gy2

    # ------------------------------------------------------------------ #
    # Lógica de posesión
    # ------------------------------------------------------------------ #
    def _update_possession_cache(
        self,
        frame: int,
        ball_bbox: Optional[list[float]],
        players: List[PlayerState],
    ) -> None:
        """
        Guarda el jugador más cercano al balón EN CADA FRAME.
        """
        if ball_bbox is None:
            return

        best_pid: Optional[int] = None
        best_dist = float("inf")

        for p in players:
            pbbox = p.get_bbox()
            if not pbbox:
                continue

            d = self._distance_bbox(ball_bbox, pbbox)
            if d < best_dist:
                best_dist = d
                best_pid = int(f'{p.player_id}')

        if best_pid is not None and best_dist <= self.max_assign_distance:
            self._possession_history.append(
                BallPossessionSnapshot(
                    frame=frame,
                    player_id=best_pid,
                    distance=best_dist,
                )
            )

    def _most_likely_scorer(self) -> Optional[int]:
        """
        Devuelve el player_id más frecuente en la ventana temporal.
        Desempata por menor distancia promedio.
        """
        if not self._possession_history:
            return None

        counter: Dict[int, int] = {}
        for snap in self._possession_history:
            counter[snap.player_id] = counter.get(snap.player_id, 0) + 1

        best_id = max(
            counter,
            key=lambda pid: (
                counter[pid],
                -np.mean(
                    [
                        s.distance
                        for s in self._possession_history
                        if s.player_id == pid
                    ]
                ),
            ),
        )
        return best_id

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: sv.Detections,
        match_id: int,
        frame_num: int,
        db: Session,
    ) -> Tuple[bool, Optional[int]]:
        """
        Returns (goal_detected, scorer_player_id)
        """

        if detections is None or len(detections) == 0:
            return False, None

        class_names = detections.data.get("class_name", [])
        if isinstance(class_names, np.ndarray):
            class_names = class_names.tolist()

        cls_name_to_id = {name: idx for idx, name in enumerate(class_names)}

        ball_idx = cls_name_to_id.get("soccer-ball")
        goal_idx = cls_name_to_id.get("soccer-goal")

        if ball_idx is None or goal_idx is None:
            return False, None

        mask_ball = detections.class_id == ball_idx
        mask_goal = detections.class_id == goal_idx

        if not mask_ball.any() or not mask_goal.any():
            return False, None

        best_ball = np.argmax(detections.confidence[mask_ball])
        best_goal = np.argmax(detections.confidence[mask_goal])

        ball_bbox = detections.xyxy[mask_ball][best_ball].tolist()
        goal_bbox = detections.xyxy[mask_goal][best_goal].tolist()


        players: List[PlayerState] = TrackCollectionPlayer(db).get_all_states()
        self._update_possession_cache(frame_num, ball_bbox, players)


        iou = float(
            sv.box_iou_batch(
                np.array([ball_bbox]), np.array([goal_bbox])
            )[0, 0]
        )

        inside = self._ball_inside_goal(ball_bbox, goal_bbox)
        scored = (iou > self.iou_threshold) or inside

        if not scored:
            return False, None


        scorer_id = self._most_likely_scorer()

        debug_logger.debug(
            f"[GoalDetector] Possession history size: {len(self._possession_history)}"
        )

        if scorer_id is None:
            debug_logger.debug("[GoalDetector] Goal detected but no scorer found")
            return True, None

        collection = TrackCollectionPlayer(db)
        player_row = collection.get_player(scorer_id)

        if player_row is None:
            error_logger.error(
                f"[GoalDetector] Player {scorer_id} not found in DB"
            )
            return True, None

        new_goals = (player_row.goals or 0) + 1
        collection.patch(int(f'{player_row.id}'), {"goals": new_goals})

        info_logger.info(
            f"[GoalDetector] Goal assigned to player {scorer_id} (total={new_goals})"
        )

        return True, scorer_id
