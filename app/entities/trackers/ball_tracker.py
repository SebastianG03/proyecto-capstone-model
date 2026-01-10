from asyncio.log import logger
import json
from typing import Any, Dict, Optional, override
from filterpy.kalman import KalmanFilter

import numpy as np
import supervision as sv
from ultralytics.models import YOLO

from app.entities.collections.track_collections import TrackCollectionBall
from app.entities.interfaces.tracker_base import Tracker
from sqlalchemy.orm import Session


class BallTracker(Tracker):

    def __init__(
        self,
        model: YOLO,
        max_age: int = 5,
        gate: float = 9.21,
        df: float = 1.0):
        super().__init__(model)
        self.kf = self._create_kf(df)
        self.bbox = []
        self.age = 0
        self.max_age = max_age
        self.gate = gate  # chi2 2-dof 95%
        self.owner_id: Optional[int] = None
        self.possession_time: float = 0.0
        self.last_seen_frame: int = -1

    def _create_kf(self, dt: float = 1.0) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        kf.R *= 9  # observación
        kf.Q[2:, 2:] *= 0.1  # velocidad
        kf.x = np.array([0., 0., 0., 0.])  # se inicializa en primera detección
        kf.P = np.eye(4) * 100.0
        kf.R = np.eye(2) * 9.0
        return kf
    
    @override
    def reset(self):
        """Resetea el estado interno del tracker."""
        self.kf = self._create_kf()
        self.age = 0
        self.owner_id = None
        self.possession_time = 0.0
        self.last_seen_frame = -1
    
    @override
    def get_object_tracks(
        self,
        detection_with_tracks,
        cls_names_inv,
        frame_num,
        detection_supervision,
        db: Session
    ):
        try:
            print("[BallTracker] ball get_object_tracks llamado frame ", frame_num)
            ball_coordinates = self._extract_ball(detection_with_tracks, cls_names_inv)
            print(f"[BallTracker] ball_coordinates: {ball_coordinates}")
            self._update_kf(ball_coordinates, frame_num)
            print(f"[BallTracker] Kalman state: {self.kf.x}")
            self._persist_state(frame_num, db)
            print(f"[BallTracker] Estado persistido en DB para frame {frame_num}")
            logger.debug(f"[BallTracker] frame {frame_num} owner={self.owner_id}")
        except Exception as e:
            logger.exception(f"[BallTracker] Error en get_object_tracks frame {frame_num}: {e}")
            print(f"[BallTracker] Error en get_object_tracks frame {frame_num}: {e}")
            raise e
        
    def _extract_ball(
        self,
        detections: sv.Detections,
        cls_names_inv: dict):
        try:
            print("[BallTracker] extract_ball llamado")
            if detections is None or len(detections) == 0:
                print("[BallTracker] No hay detecciones")
                return None

            print("[BallTracker] extract_ball procesando detecciones")
            print(f"[BallTracker] Clases disponibles: {cls_names_inv}")
            print(f"[BallTracker] Detecciones - class_ids: {detections.class_id}")
            print(f"[BallTracker] Detecciones - confianza: {detections.confidence}")
            xyxy = np.asarray(getattr(detections, "xyxy", None))
            print(f"[BallTracker] xyxy: {xyxy}")
            class_ids = np.asarray(getattr(detections, "class_id", None))
            print(f"[BallTracker] class_ids: {class_ids}")
            if xyxy is None or class_ids is None:
                print("[BallTracker] No xyxy o class_ids")
                return None
            print("[BallTracker] Buscando clase 'ball'")
            ball_idx = cls_names_inv.get("ball")
            if ball_idx is None:
                return None
            mask = class_ids == ball_idx
            print(f"[BallTracker] mask: {mask} resultante de class ids {class_ids} y ball idx{ball_idx}")
            if not mask.any():
                print("[BallTracker] Ningún balón detectado, intentando recuperar posición")
                return self._recover_ball_position(detections, cls_names_inv)
            
            valid_indices = np.where(mask)[0]
            conf = detections.confidence
            if len(valid_indices) > 1 and conf is not None:
                confidences = conf[mask]
                best_idx = np.argmax(confidences)
                best_detection_idx = valid_indices[best_idx]
            else:
                best_detection_idx = valid_indices[0]
            
            # Extraer coordenadas
            bbox = detections.xyxy[best_detection_idx]
            self.bbox = list(map(float, bbox))
            cx, cy = self._bbox_to_center(self.bbox)
            
            print(f"[BallTracker] Balón recuperado: centro=({cx}, {cy})")
            return (cx, cy)
        except Exception as e:
            print(f"[BallTracker] Error extrayendo balón: {e}")
            raise e

    def _update_kf(
        self,
        ball_coordinates: Optional[tuple[float, float]],
        frame_number: int):
        if ball_coordinates is not None:
            z = np.array(ball_coordinates)
            
            if np.allclose(self.kf.x, 0):
                self.kf.x = np.array([z[0], z[1], 0., 0.])
            
            y, S = self.kf.y, self.kf.S
            det = np.linalg.det(S)
            
            if det < 1e-6:
                logger.warning("Matriz S casi singular, saltando actualización")
                S_inv = np.linalg.pinv(S)
            else:
                S_inv = np.linalg.inv(S)
            
            d2 = float((y.T @ S_inv @ y).item())
            if d2 < self.gate:
                self.kf.update(z)
                self.age = 0
                self.last_seen_frame = frame_number
            else:
                logger.debug(f"Detección rechazada por gate (d2={d2:.2f})")
        else:
            self.age += 1
        self.kf.predict()

    def get_ball_state(self) -> Optional[Dict[str, Any]]:
        if self.age > self.max_age:
            return None
        logger.debug(f"Recuperando Kalman {self.kf.x} con longitud {len(self.kf.x)}")
        x, y, _, _ = self.kf.x
        return {
            "x": float(x),
            "y": float(y),
            "vx": float(self.kf.x[2]),
            "vy": float(self.kf.x[3]),
            "bbox": self.bbox,
            "owner_id": self.owner_id,
            "possession_time": self.possession_time,
            "last_seen_frame": self.last_seen_frame
        }
    
    def _recover_ball_position(
    self,
    detections: sv.Detections,
    cls_names_inv: dict) -> Optional[tuple[float, float]]:
        """
        Recupera la posición del balón usando heurísticas cuando no es detectado.
        """
        print("[BallTracker] Aplicando estrategias de recuperación...")
        
        # Estrategia 1: Usar la última posición conocida si el Kalman filter es confiable
        if self.age < self.max_age and not np.allclose(self.kf.x[:2], 0):
            predicted_x, predicted_y = self.kf.x[0], self.kf.x[1]
            print(f"[BallTracker] Usando predicción de Kalman: ({predicted_x}, {predicted_y})")
            return (predicted_x, predicted_y)
        
        # Estrategia 2: Buscar en regiones cercanas a jugadores (el balón suele estar cerca)
        player_idx = cls_names_inv.get("player")
        if player_idx is not None:
            player_mask = detections.class_id == player_idx
            if player_mask.any():
                player_bboxes = detections.xyxy[player_mask]
                player_centers = []
                for bbox in player_bboxes:
                    cx, cy = self._bbox_to_center(bbox)
                    player_centers.append((cx, cy))
                
                # Estimar posición del balón basándose en jugadores
                if player_centers:
                    # Simple heurística: balón cerca del centro del campo
                    avg_x = np.mean([c[0] for c in player_centers])
                    avg_y = np.mean([c[1] for c in player_centers])
                    
                    # Ajustar posición basándose en la dirección del juego
                    if len(player_centers) > 1:
                        # Usar vector de movimiento de jugadores
                        direction_x = player_centers[-1][0] - player_centers[0][0]
                        direction_y = player_centers[-1][1] - player_centers[0][1]
                        
                        # Posicionar balón ligeramente adelante del promedio
                        ball_x = avg_x + direction_x * 0.2
                        ball_y = avg_y + direction_y * 0.2
                    else:
                        ball_x, ball_y = avg_x, avg_y
                    
                    print(f"[BallTracker] Balón estimado por posición de jugadores: ({ball_x}, {ball_y})")
                    return (float(ball_x), float(ball_y))
        
        # Estrategia 3: No se puede recuperar
        print("[BallTracker] No se pudo recuperar la posición del balón")
        return None

    def _persist_state(self, frame_number: int, db: Session):
        state = self.get_ball_state()
        if state is None:
            return
        tracks_collection = TrackCollectionBall(db)
        print("Actual state to persist: ", state)
        payload = {
            "frame_index": frame_number,
            "x": state["x"],
            "y": state["y"],
            "z": 0.0,
            "vx": state["vx"],
            "vy": state["vy"],
            "bbox": json.dumps(state["bbox"]),
            "owner_id": state["owner_id"],
            "track_id": 0
        }
        try:
            print("Creating new ball record: ", payload)
            tracks_collection.post(payload)
        except Exception as e:
            logger.exception(f"Error al guardar el balon en frame {frame_number}: {e}")
            raise e
