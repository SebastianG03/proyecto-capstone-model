from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import supervision as sv
from sqlalchemy.orm import Session

from app.core.config import BATCH_SIZE
from app.entities.collections import TrackCollectionPlayer
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import PlayerState
from app.entities.models.detected_object_data import AnalysisData
from app.logger import debug_logger, error_logger, info_logger
import app.entities.utils.tools_context as context 
import app.entities.utils.global_values_store as value_store

class ShotOutcome(Enum):
    GOAL = "goal"
    SAVED = "saved"  # Atajado por el portero
    MISSED = "missed"  # Desviado
    BLOCKED = "blocked"  # Bloqueado por defensa
    UNKNOWN = "unknown"


@dataclass
class BallPossessionSnapshot:
    frame: int
    player_id: int
    distance: float


@dataclass
class BallTrajectoryPoint:
    frame: int
    position: Tuple[float, float]  # centro del balon
    velocity: Optional[Tuple[float, float]] = None


@dataclass
class ShotEvent:
    frame: int
    player_id: int
    outcome: ShotOutcome
    ball_speed: float
    distance_to_goal: float
    goal_bbox: List[float]
    confidence: float = 0.0

@dataclass
class Goal:
    frame: int
    goal_id: int
    bbox: list[float]
    confidence: float


class ShotDetector:
    """
    Detecta TIROS AL ARCO (no solo goles) y asigna al jugador que realizo el tiro.
    
    Logica:
    1. Detecta cuando el balon se mueve rapidamente hacia el arco
    2. Determina si fue un tiro (vs pase) basado en velocidad y direccion
    3. Registra el jugador que tuvo posesion justo antes del tiro
    4. Clasifica el resultado: gol, atajado, desviado, etc.
    """

    # Ventana de posesion para identificar al tirador (frames antes del tiro)
    POSSESSION_WINDOW = 15
    
    # Ventana de trayectoria para analizar direccion del balon
    TRAJECTORY_WINDOW = 10
    
    # Umbral de velocidad minima para considerar un "tiro" (pixels/frame)
    MIN_SHOT_SPEED = 15.0
    
    # Umbral de velocidad para considerar un "potente" (ayuda a distinguir de pases)
    HIGH_SHOT_SPEED = 25.0
    
    # Umbral de proximidad al arco para considerar "tiro al arco" (pixels)
    GOAL_PROXIMITY_THRESHOLD = 300.0
    
    # Frames de cooldown entre tiros detectados (evitar duplicados)
    SHOT_COOLDOWN_FRAMES = 30
    
    # Umbral de confianza para deteccion de balon/arco
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        iou_threshold: float = 0.3,
        pixel_threshold: float = 200.0,
        max_assign_distance: float = 170.0,
        goal_zone_expansion: float = 50.0,  # Expandir zona de gol para detectar tiros cercanos
    ):
        self.iou_threshold = iou_threshold
        self.pixel_threshold = pixel_threshold
        self.max_assign_distance = max_assign_distance
        self.goal_zone_expansion = goal_zone_expansion
        self.target_name = "soccer-goal"

        # Historial de posesion
        self._possession_history: deque[BallPossessionSnapshot] = deque(
            maxlen=self.POSSESSION_WINDOW
        ) # limitar memoria
        
        # Trayectoria del balon
        self._ball_trajectory: deque[BallTrajectoryPoint] = deque(
            maxlen=self.TRAJECTORY_WINDOW
        ) # limitar memoria
        
        # Registro de tiros detectados
        self._shots: List[ShotEvent] = [] # limitar memoria
        
        # Control de cooldown
        self._last_shot_frame: int = -self.SHOT_COOLDOWN_FRAMES # limitar memoria
        
        # Estado anterior para calcular velocidad
        self._prev_ball_pos: Optional[Tuple[float, float]] = None # limitar memoria
        self._prev_frame: int = 0


    @staticmethod
    def _bbox_center(bbox: list[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _distance_point_to_bbox(point: Tuple[float, float], bbox: list[float]) -> float:
        """Distancia desde un punto al centro de un bbox"""
        cx, cy = ShotDetector._bbox_center(bbox)
        return float(np.linalg.norm(np.array(point) - np.array([cx, cy])))

    @staticmethod
    def _distance_bbox(bbox_a: list[float], bbox_b: list[float]) -> float:
        return float(np.linalg.norm(
            np.array(ShotDetector._bbox_center(bbox_a))
            - np.array(ShotDetector._bbox_center(bbox_b))
        ))

    @staticmethod
    def _ball_inside_goal(ball_bbox: list[float], goal_bbox: list[float]) -> bool:
        bx1, by1, bx2, by2 = ball_bbox
        gx1, gy1, gx2, gy2 = goal_bbox
        return bx1 >= gx1 and bx2 <= gx2 and by1 >= gy1 and by2 <= gy2

    @staticmethod
    def _expand_bbox(bbox: list[float], expansion: float) -> list[float]:
        """Expande un bbox por cierto margen"""
        x1, y1, x2, y2 = bbox
        return [x1 - expansion, y1 - expansion, x2 + expansion, y2 + expansion]

    @staticmethod
    def _calculate_direction(trajectory: deque[BallTrajectoryPoint]) -> Optional[Tuple[float, float]]:
        """Calcula la direccion promedio del movimiento del balon"""
        if len(trajectory) < 3:
            return None
        
        # Usar regresion lineal simple para direccion
        points = list(trajectory)
        x_coords = [p.position[0] for p in points]
        y_coords = [p.position[1] for p in points]
        
        # Vector de direccion (ultimo punto - primer punto)
        dx = x_coords[-1] - x_coords[0]
        dy = y_coords[-1] - y_coords[0]
        
        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            return None
            
        return (dx / norm, dy / norm)


    def _update_possession_cache(
        self,
        frame: int,
        ball_bbox: Optional[list[float]],
        players: List[PlayerState],
    ) -> Optional[int]:
        """
        Guarda el jugador mas cercano al balon EN CADA FRAME.
        Retorna el player_id del mejor candidato o None.
        """
        if ball_bbox is None:
            return None

        best_pid: Optional[int] = None
        best_dist = float("inf")

        for player in players:
            player_bbox = player.get_bbox()
            if not player_bbox:
                continue

            distance = self._distance_bbox(ball_bbox, player_bbox)
            if distance < best_dist:
                best_dist = distance
                best_pid = int(f"{player.player_id}")

        if best_pid is not None and best_dist <= self.max_assign_distance:
            self._possession_history.append(
                BallPossessionSnapshot(
                    frame=frame,
                    player_id=best_pid,
                    distance=best_dist,
                )
            )
            return best_pid
        return None

    def _most_likely_shooter(self) -> Optional[int]:
        """
        Devuelve el player_id mas frecuente en la ventana de posesion.
        Este es el jugador que probablemente realizo el tiro.
        """
        if not self._possession_history:
            return None

        counter: Dict[int, int] = {}
        for snap in self._possession_history:
            counter[snap.player_id] = counter.get(snap.player_id, 0) + 1

        # Desempatar por frecuencia y luego por distancia promedio (menor es mejor)
        best_id = max(
            counter,
            key=lambda pid: (
                counter[pid],
                -np.mean([
                    s.distance for s in self._possession_history if s.player_id == pid
                ]),
            ),
        )
        return best_id
    
    def _get_last_ball_instances(self, actual_frame: int, intances: list[BallEventModel]) -> list[BallEventModel]:
        return [b for b in intances if int(f'{b.frame_index}') <= actual_frame and int(f'{b.frame_index}') >= actual_frame - BATCH_SIZE]


    def _update_ball_trajectory(
        self, 
        frame: int, 
        ball_bbox: Optional[list[float]]
    ) -> None:
        """Actualiza el historial de posiciones del balon"""
        if ball_bbox is None:
            return
            
        center = self._bbox_center(ball_bbox)
        
        # Calcular velocidad si tenemos posicion anterior
        velocity = None
        if self._prev_ball_pos is not None and frame > self._prev_frame:
            dt = frame - self._prev_frame
            vx = (center[0] - self._prev_ball_pos[0]) / dt
            vy = (center[1] - self._prev_ball_pos[1]) / dt
            velocity = (vx, vy)
        
        self._ball_trajectory.append(
            BallTrajectoryPoint(frame=frame, position=center, velocity=velocity)
        )
        
        self._prev_ball_pos = center
        self._prev_frame = frame

    def _is_moving_towards_goal(
        self, 
        goal_bbox: list[float]
    ) -> Tuple[bool, float, float]:
        """
        Determina si el balon se esta moviendo hacia el arco.
        Retorna: (es_hacia_arco, velocidad, distancia_al_arco)
        """
        if len(self._ball_trajectory) < 3:
            return False, 0.0, float('inf')
        
        # Calcular velocidad actual
        recent_points = list(self._ball_trajectory)[-3:]
        if recent_points[-1].velocity is None:
            return False, 0.0, float('inf')
        
        vx, vy = recent_points[-1].velocity
        speed = np.sqrt(vx**2 + vy**2)
        
        # Posicion actual y direccion
        current_pos = recent_points[-1].position
        goal_center = self._bbox_center(goal_bbox)
        
        # Vector hacia el arco
        dx = goal_center[0] - current_pos[0]
        dy = goal_center[1] - current_pos[1]
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        
        if dist_to_goal == 0:
            return False, speed, 0.0
        
        # Normalizar vector hacia arco
        dx_norm = dx / dist_to_goal
        dy_norm = dy / dist_to_goal
        
        # Producto punto: si es positivo, va hacia el arco
        direction = self._calculate_direction(self._ball_trajectory)
        if direction is None:
            return False, speed, dist_to_goal
        
        dot_product = direction[0] * dx_norm + direction[1] * dy_norm
        
        # Considerar "hacia arco" si el producto punto es > 0.5 (angulo < 60 grados)
        is_towards = dot_product > 0.5 and speed > self.MIN_SHOT_SPEED
        
        return is_towards, speed, dist_to_goal

    def _determine_shot_outcome(
        self,
        ball_bbox: list[float],
        goal_bbox: list[float],
        players: list[PlayerState],
    ) -> ShotOutcome:
        """
        Determina el resultado del tiro: gol, atajado, desviado, etc.
        """
        if self._ball_inside_goal(ball_bbox, goal_bbox):
            return ShotOutcome.GOAL

        ball_center = self._bbox_center(ball_bbox)
        goal_center = self._bbox_center(goal_bbox)
        dist_to_goal = np.linalg.norm(np.array(ball_center) - np.array(goal_center))
        
        if dist_to_goal < 150:
            player_idx = None
            for player in players:
                bbox = player.get_bbox()
                if bbox is None:
                    continue

                distance = self._distance_bbox(ball_bbox, bbox)
                if distance < 100:
                    player_idx = int(f"{player.player_id}")
                    break
            
            if player_idx is not None:
                return ShotOutcome.BLOCKED
        
        return ShotOutcome.MISSED

    def update(
        self,
        detections: sv.Detections,
        match_id: int,
        frame_num: int,
        db: Session,
    ) -> Tuple[bool, Optional[ShotEvent]]:
        
        """
        Detecta tiros al arco.
        
        Returns:
            (shot_detected, shot_event)
            shot_event es None si no se detecto tiro
        """
        
        if detections is None or len(detections) == 0:
            return False, None
        
        mask = detections.class_id == self.target_name
        bboxes = np.asarray(detections.xyxy)    
        goal_ids = np.asarray(detections.tracker_id) 
        confidence = np.asarray(detections.confidence)
        
        if not mask.any() or not bboxes.any() or not goal_ids.any() or not confidence.any():
            return False, None
        
        bboxes = bboxes[mask]
        goal_ids = goal_ids[mask]
        confidence = confidence[mask]
        info_logger.info(f"[ShotDetector] Bboxes: {bboxes}")
        info_logger.info(f"[ShotDetector] Goal IDs: {goal_ids}")
        info_logger.info(f"[ShotDetector] Mask: {mask}")
        info_logger.info(f"[ShotDetector] Data: {detections.data}")
        
        goal = Goal(
            frame=frame_num,
            bbox=[0,0,0,0],
            confidence=0,
            goal_id=1
        )
        
        if not bboxes.any() and not goal_ids.any():
            return False, None

        for bbox, id, conf in zip(bboxes, goal_ids, confidence):
            if id is None or not str(id).isnumeric():
                id = 1
            
            if bbox is None:
                continue
            
            bbox_list = bbox.tolist()
            x, y = self._bbox_center(bbox_list)
            
            if goal.confidence < conf:
                goal.confidence = conf
                goal.bbox = bbox_list
                goal.goal_id = int(id)
            
            context.analysis_context.tools.analysis_data_collector.add_row(AnalysisData(
                    frame=frame_num,
                    track_id=1,
                    x=x,
                    y=y,
                    vclass="goal",
                    timestamps=value_store.globals.timestamp
                ))

        ball_instances = context.analysis_context.tools.ball_records.get_balls_last_frames(frame_num)
        ball_instances = self._get_last_ball_instances(frame_num, ball_instances)
        info_logger.info(f"[ShotDetector] Ball instances: {len(ball_instances)}")

        if len(ball_instances) == 0:
            return False, None
    
        ball_bbox = ball_instances[0].get_bbox()
        
        if ball_bbox is None:
            for ball in ball_instances:
                if ball.get_bbox() is not None:
                    ball_bbox = ball.get_bbox()
                    break
        
        if ball_bbox is None:
            return False, None
        
        self._update_ball_trajectory(frame_num, ball_bbox)

        # Obtener jugadores y actualizar posesion
        players: List[PlayerState] = context.analysis_context.tools.player_records.get_all_states(frame_index=frame_num)
        # Todo Error el ultimo bbox no siempre va a tener el balon, hay que cambiarlo para verificar una lista de bbox
        current_possession = self._update_possession_cache(frame_num, ball_bbox, players)
        goal_bbox = goal.bbox

        # Verificar si el balon se mueve hacia el arco
        # Todo Error el ultimo bbox no siempre va a tener el balon, es necesario considerar todos los bbox, esta logica sera pasada a Rust
        is_towards_goal, speed, dist_to_goal = self._is_moving_towards_goal(goal_bbox)
        
        if not is_towards_goal:
            return False, None
        
        # Verificar cooldown (evitar detectar el mismo tiro multiples veces)
        if frame_num - self._last_shot_frame < self.SHOT_COOLDOWN_FRAMES:
            return False, None
        
        # Verificar que este dentro de rango de tiro al arco
        if dist_to_goal > self.GOAL_PROXIMITY_THRESHOLD:
            return False, None

        # Detectar tiro al arco!
        self._last_shot_frame = frame_num
        
        # Determinar quien tiro
        shooter_id = self._most_likely_shooter()
        
        # Determinar resultado
        outcome = self._determine_shot_outcome(ball_bbox, goal_bbox, players)
        
        # Crear evento de tiro
        # Se considera que la confianza del balon en el momento de la deteccion es 0.47 dado que no se almacena la confianza en base a pruebas, esa es la confianza maxima con la que se detecta en el arco, siendo la minima 0.35 
        shot_event = ShotEvent(
            frame=frame_num,
            player_id=shooter_id if shooter_id else -1,
            outcome=outcome,
            ball_speed=speed,
            distance_to_goal=dist_to_goal,
            goal_bbox=goal_bbox,
            confidence=(0.47 + goal.confidence) / 2
        )
        
        self._shots.append(shot_event)
        
        info_logger.info(
            f"[ShotDetector] ¡TIRO DETECTADO! "
            f"Frame={frame_num}, "
            f"Jugador={shooter_id}, "
            f"Resultado={outcome.value}, "
            f"Velocidad={speed:.1f}px/f, "
            f"Distancia={dist_to_goal:.1f}px"
        )

        if shooter_id is not None:
            self._update_player_stats(shooter_id, outcome, db)

        return True, shot_event

    def _update_player_stats(self, player_id: int, outcome: ShotOutcome, db: Session) -> None:
        """Actualiza las estadisticas de tiros del jugador en la base de datos"""
        try:
            collection = TrackCollectionPlayer()
            player_row = collection.get_player(player_id)

            if player_row is None:
                error_logger.error(f"[ShotDetector] Player {player_id} not found in DB")
                return

            current_goals = getattr(player_row, 'goals', 0) or 0
            
            new_goals = current_goals + (1 if outcome == ShotOutcome.GOAL else 0)

            updates = {
                'goals': new_goals
            }
            
            collection.patch(int(f"{player_row.player_id}"), updates)

            info_logger.info(
                f"[ShotDetector] Stats updated for player {player_id}: "
                f"goals={new_goals}"
            )

        except Exception as e:
            error_logger.error(f"[ShotDetector] Error updating player stats: {e}")

    def get_shots(self) -> List[ShotEvent]:
        """Retorna todos los tiros detectados"""
        return self._shots.copy()

    def reset(self) -> None:
        """Reinicia el estado del detector"""
        self._possession_history.clear()
        self._ball_trajectory.clear()
        self._shots.clear()
        self._last_shot_frame = -self.SHOT_COOLDOWN_FRAMES
        self._prev_ball_pos = None
        self._prev_frame = 0



class GoalScorerDetector(ShotDetector):
    """
    Version legacy que mantiene la interfaz anterior pero usa ShotDetector internamente.
    Solo detecta GOLES, no todos los tiros.
    """
    
    def updateScorer(
        self,
        detections: sv.Detections,
        match_id: int,
        frame_num: int,
        db: Session,
    ) -> Tuple[bool, Optional[int]]:
        """
        Interfaz compatible con version anterior.
        Returns (goal_detected, scorer_player_id)
        """
        shot_detected, shot_event = super().update(detections, match_id, frame_num, db)
        
        if shot_detected and shot_event and shot_event.outcome == ShotOutcome.GOAL:
            return True, shot_event.player_id if shot_event.player_id != -1 else None
        
        return False, None