from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from polars import override
import supervision as sv
from sqlalchemy.orm import Session

from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState
from app.logger import debug_logger, error_logger, info_logger


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
    position: Tuple[float, float]  # centro del balón
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


class ShotDetector:
    """
    Detecta TIROS AL ARCO (no solo goles) y asigna al jugador que realizó el tiro.
    
    Lógica:
    1. Detecta cuando el balón se mueve rápidamente hacia el arco
    2. Determina si fue un tiro (vs pase) basado en velocidad y dirección
    3. Registra el jugador que tuvo posesión justo antes del tiro
    4. Clasifica el resultado: gol, atajado, desviado, etc.
    """

    # Ventana de posesión para identificar al tirador (frames antes del tiro)
    POSSESSION_WINDOW = 15
    
    # Ventana de trayectoria para analizar dirección del balón
    TRAJECTORY_WINDOW = 10
    
    # Umbral de velocidad mínima para considerar un "tiro" (pixels/frame)
    MIN_SHOT_SPEED = 15.0
    
    # Umbral de velocidad para considerar un "potente" (ayuda a distinguir de pases)
    HIGH_SHOT_SPEED = 25.0
    
    # Umbral de proximidad al arco para considerar "tiro al arco" (pixels)
    GOAL_PROXIMITY_THRESHOLD = 300.0
    
    # Frames de cooldown entre tiros detectados (evitar duplicados)
    SHOT_COOLDOWN_FRAMES = 30
    
    # Umbral de confianza para detección de balón/arco
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

        # Historial de posesión
        self._possession_history: deque[BallPossessionSnapshot] = deque(
            maxlen=self.POSSESSION_WINDOW
        )
        
        # Trayectoria del balón
        self._ball_trajectory: deque[BallTrajectoryPoint] = deque(
            maxlen=self.TRAJECTORY_WINDOW
        )
        
        # Registro de tiros detectados
        self._shots: List[ShotEvent] = []
        
        # Control de cooldown
        self._last_shot_frame: int = -self.SHOT_COOLDOWN_FRAMES
        
        # Estado anterior para calcular velocidad
        self._prev_ball_pos: Optional[Tuple[float, float]] = None
        self._prev_frame: int = 0

    # ------------------------------------------------------------------ #
    # Helpers geométricos
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bbox_center(bbox: list[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _distance_point_to_bbox(point: Tuple[float, float], bbox: list[float]) -> float:
        """Distancia desde un punto al centro de un bbox"""
        cx, cy = ShotDetector._bbox_center(bbox)
        return np.linalg.norm(np.array(point) - np.array([cx, cy]))

    @staticmethod
    def _distance_bbox(bbox_a: list[float], bbox_b: list[float]) -> float:
        return np.linalg.norm(
            np.array(ShotDetector._bbox_center(bbox_a))
            - np.array(ShotDetector._bbox_center(bbox_b))
        )

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
        """Calcula la dirección promedio del movimiento del balón"""
        if len(trajectory) < 3:
            return None
        
        # Usar regresión lineal simple para dirección
        points = list(trajectory)
        x_coords = [p.position[0] for p in points]
        y_coords = [p.position[1] for p in points]
        
        # Vector de dirección (último punto - primer punto)
        dx = x_coords[-1] - x_coords[0]
        dy = y_coords[-1] - y_coords[0]
        
        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            return None
            
        return (dx / norm, dy / norm)

    # ------------------------------------------------------------------ #
    # Lógica de posesión
    # ------------------------------------------------------------------ #
    def _update_possession_cache(
        self,
        frame: int,
        ball_bbox: Optional[list[float]],
        players: List[PlayerState],
    ) -> Optional[int]:
        """
        Guarda el jugador más cercano al balón EN CADA FRAME.
        Retorna el player_id del mejor candidato o None.
        """
        if ball_bbox is None:
            return None

        best_pid: Optional[int] = None
        best_dist = float("inf")

        for p in players:
            pbbox = p.get_bbox()
            if not pbbox:
                continue

            d = self._distance_bbox(ball_bbox, pbbox)
            if d < best_dist:
                best_dist = d
                best_pid = int(f"{p.player_id}")

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
        Devuelve el player_id más frecuente en la ventana de posesión.
        Este es el jugador que probablemente realizó el tiro.
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

    # ------------------------------------------------------------------ #
    # Lógica de trayectoria
    # ------------------------------------------------------------------ #
    def _update_ball_trajectory(
        self, 
        frame: int, 
        ball_bbox: Optional[list[float]]
    ) -> None:
        """Actualiza el historial de posiciones del balón"""
        if ball_bbox is None:
            return
            
        center = self._bbox_center(ball_bbox)
        
        # Calcular velocidad si tenemos posición anterior
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
        Determina si el balón se está moviendo hacia el arco.
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
        
        # Posición actual y dirección
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
        
        # Considerar "hacia arco" si el producto punto es > 0.5 (ángulo < 60 grados)
        is_towards = dot_product > 0.5 and speed > self.MIN_SHOT_SPEED
        
        return is_towards, speed, dist_to_goal

    def _determine_shot_outcome(
        self,
        ball_bbox: list[float],
        goal_bbox: list[float],
        detections: sv.Detections,
        class_names: List[str]
    ) -> ShotOutcome:
        """
        Determina el resultado del tiro: gol, atajado, desviado, etc.
        """
        # Verificar si es gol (balón dentro del arco)
        if self._ball_inside_goal(ball_bbox, goal_bbox):
            return ShotOutcome.GOAL
        
        # Verificar si hay portero cerca (posible atajada)
        # Nota: Asume que tienes una clase 'goalkeeper' o similar en tu modelo
        goalkeeper_idx = None
        for idx, name in enumerate(class_names):
            if 'keeper' in name.lower() or 'portero' in name.lower() or 'goalkeeper' in name.lower():
                goalkeeper_idx = idx
                break
        
        if goalkeeper_idx is not None:
            mask_keeper = detections.class_id == goalkeeper_idx
            if mask_keeper.any():
                keepers = detections.xyxy[mask_keeper]
                for keeper_bbox in keepers:
                    # Si el balón está cerca del portero y no entró, probable atajada
                    if self._distance_bbox(ball_bbox, keeper_bbox.tolist()) < 100:
                        return ShotOutcome.SAVED
        
        # Si el balón pasó cerca del arco pero no entró
        ball_center = self._bbox_center(ball_bbox)
        goal_center = self._bbox_center(goal_bbox)
        dist_to_goal = np.linalg.norm(np.array(ball_center) - np.array(goal_center))
        
        # Si está cerca del arco pero no entró, podría ser bloqueado o desviado
        if dist_to_goal < 150:  # Muy cerca del arco
            # Verificar si hay jugadores cercanos (posible bloqueo)
            player_idx = None
            for idx, name in enumerate(class_names):
                if 'player' in name.lower():
                    player_idx = idx
                    break
            
            if player_idx is not None:
                mask_players = detections.class_id == player_idx
                if mask_players.any():
                    players = detections.xyxy[mask_players]
                    for player_bbox in players:
                        if self._distance_bbox(ball_bbox, player_bbox.tolist()) < 80:
                            return ShotOutcome.BLOCKED
        
        # Por defecto, desviado si no es gol
        return ShotOutcome.MISSED

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
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
            shot_event es None si no se detectó tiro
        """
        
        if detections is None or len(detections) == 0:
            return False, None

        class_names = detections.data.get("class_name", [])
        if isinstance(class_names, np.ndarray):
            class_names = class_names.tolist()

        # Buscar índices de clases
        cls_name_to_id = {name: idx for idx, name in enumerate(class_names)}
        
        ball_idx = cls_name_to_id.get("soccer-ball")
        goal_idx = cls_name_to_id.get("soccer-goal")

        if ball_idx is None or goal_idx is None:
            info_logger.debug("[ShotDetector] Ball or goal class not found")
            return False, None

        mask_ball = detections.class_id == ball_idx
        mask_goal = detections.class_id == goal_idx

        if not mask_ball.any() or not mask_goal.any():
            return False, None

        # Seleccionar detecciones con mayor confianza
        best_ball = np.argmax(detections.confidence[mask_ball])
        best_goal = np.argmax(detections.confidence[mask_goal])

        ball_bbox = detections.xyxy[mask_ball][best_ball].tolist()
        goal_bbox = detections.xyxy[mask_goal][best_goal].tolist()
        
        ball_conf = float(detections.confidence[mask_ball][best_ball])
        goal_conf = float(detections.confidence[mask_goal][best_goal])

        # Actualizar trayectoria del balón
        self._update_ball_trajectory(frame_num, ball_bbox)

        # Obtener jugadores y actualizar posesión
        players: List[PlayerState] = TrackCollectionPlayer(db).get_all_states()
        current_possession = self._update_possession_cache(frame_num, ball_bbox, players)

        # Verificar si el balón se mueve hacia el arco
        is_towards_goal, speed, dist_to_goal = self._is_moving_towards_goal(goal_bbox)
        
        if not is_towards_goal:
            return False, None
        
        # Verificar cooldown (evitar detectar el mismo tiro múltiples veces)
        if frame_num - self._last_shot_frame < self.SHOT_COOLDOWN_FRAMES:
            return False, None
        
        # Verificar que esté dentro de rango de tiro al arco
        if dist_to_goal > self.GOAL_PROXIMITY_THRESHOLD:
            return False, None

        # Detectar tiro al arco!
        self._last_shot_frame = frame_num
        
        # Determinar quién tiró
        shooter_id = self._most_likely_shooter()
        
        # Determinar resultado
        outcome = self._determine_shot_outcome(ball_bbox, goal_bbox, detections, class_names)
        
        # Crear evento de tiro
        shot_event = ShotEvent(
            frame=frame_num,
            player_id=shooter_id if shooter_id else -1,
            outcome=outcome,
            ball_speed=speed,
            distance_to_goal=dist_to_goal,
            goal_bbox=goal_bbox,
            confidence=(ball_conf + goal_conf) / 2
        )
        
        self._shots.append(shot_event)
        
        # Log detallado
        info_logger.info(
            f"[ShotDetector] ¡TIRO DETECTADO! "
            f"Frame={frame_num}, "
            f"Jugador={shooter_id}, "
            f"Resultado={outcome.value}, "
            f"Velocidad={speed:.1f}px/f, "
            f"Distancia={dist_to_goal:.1f}px"
        )

        # Actualizar estadísticas del jugador en BD
        if shooter_id is not None:
            self._update_player_stats(shooter_id, outcome, db)

        return True, shot_event

    def _update_player_stats(self, player_id: int, outcome: ShotOutcome, db: Session) -> None:
        """Actualiza las estadísticas de tiros del jugador en la base de datos"""
        try:
            collection = TrackCollectionPlayer(db)
            player_row = collection.get_player(player_id)

            if player_row is None:
                error_logger.error(f"[ShotDetector] Player {player_id} not found in DB")
                return

            # Actualizar contadores
            current_shots = getattr(player_row, 'shots', 0) or 0
            current_goals = getattr(player_row, 'goals', 0) or 0
            current_on_target = getattr(player_row, 'shots_on_target', 0) or 0

            new_shots = current_shots + 1
            
            # Determinar si fue a puerta (goal o saved = a puerta)
            on_target = outcome in [ShotOutcome.GOAL, ShotOutcome.SAVED]
            new_on_target = current_on_target + (1 if on_target else 0)
            
            # Si fue gol, actualizar goles
            new_goals = current_goals + (1 if outcome == ShotOutcome.GOAL else 0)

            updates = {
                'shots': new_shots,
                'shots_on_target': new_on_target,
                'goals': new_goals
            }
            
            collection.patch(int(f"{player_row.id}"), updates)

            info_logger.info(
                f"[ShotDetector] Stats updated for player {player_id}: "
                f"shots={new_shots}, on_target={new_on_target}, goals={new_goals}"
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


# ------------------------------------------------------------------------- #
# Mantener GoalScorerDetector para compatibilidad hacia atrás
# ------------------------------------------------------------------------- #

class GoalScorerDetector(ShotDetector):
    """
    Versión legacy que mantiene la interfaz anterior pero usa ShotDetector internamente.
    Solo detecta GOLES, no todos los tiros.
    """
    
    def update(
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