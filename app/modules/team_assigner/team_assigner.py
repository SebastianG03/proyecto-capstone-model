import json
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from cv2.typing import MatLike
from sqlalchemy.orm import Session

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from app.entities.collections.track_collections import TrackCollectionPlayer
from app.entities.models.PlayerState import PlayerStateModel
from app.entities.utils.singleton import Singleton
from app.logger import *

class TeamAssigner(metaclass=Singleton):
    """
    Versión optimizada para uso frame-a-frame sin renombrar la clase.
    - Bootstrap (one-shot) de colores de equipo.
    - Clasificación O(1) por jugador por frame.
    - Smoothing temporal por jugador (ventana configurable).
    - Filtrado HSV para reducir césped/ruido; toma torso central.
    - Fallbacks robustos si bbox está recortado o inválido.
    """

    def __init__(
        self,
        smoothing_window: int = 11,
        min_bootstrap_players: int = 8,
        torso_fraction: float = 0.4,
        hsv_green_thresh: Tuple[int, int, int] = (35, 40, 40),
    ):
        # estado del modelo
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.team_colors: Dict[int, np.ndarray] = {}

        # cache y smoothing
        self.player_team_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )
        self.player_team_cache: Dict[int, int] = {}  # última decisión estable

        # parámetros
        self.smoothing_window = smoothing_window
        self.min_bootstrap_players = min_bootstrap_players
        self.torso_fraction = float(np.clip(torso_fraction, 0.2, 0.6))
        self.hsv_green_thresh = hsv_green_thresh  # (h,s,v) baseline to ignore greens

    # ---------------------------
    # Helpers bbox / crop
    # ---------------------------
    def _coords_from_bbox(self, frame: MatLike, bbox: List[int]) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w - 1, int(bbox[2]))
        y2 = min(h - 1, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _safe_crop(self, frame: MatLike, bbox: List[int]) -> Optional[np.ndarray]:
        coords = self._coords_from_bbox(frame, bbox)
        if not coords:
            return None
        x1, y1, x2, y2 = coords
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _torso_region(self, crop: np.ndarray) -> Optional[np.ndarray]:
        h = crop.shape[0]
        start = int(h * (0.2))  # ignorar la cabeza ~20%
        end = int(h * (0.2 + self.torso_fraction))
        if end <= start:
            return None
        return crop[start:end, :]

    # ---------------------------
    # HSV filtering (excluir césped)
    # ---------------------------
    def _mask_non_green(self, img_bgr: np.ndarray) -> np.ndarray:
        """Devuelve máscara booleana de píxeles NO-verdes (True = candidato)."""
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = self.hsv_green_thresh
        lower_green = np.array([30, 40, 30])  # tolerancia base (puedes ajustar)
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        return green_mask == 0  # True para no-verdes

    # ---------------------------
    # Extraer color dominante (rápido + robusto)
    # ---------------------------
    def extract_player_color(self, frame: MatLike, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Intenta devolver un arreglo RGB (float) representativo del torso.
        Aplica: crop seguro → torso → máscara no-verde → sample de píxeles → centroid simple.
        Evita KMeans por jugador para agilizar (salvo en bootstrap).
        """
        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None

        torso = self._torso_region(crop)
        if torso is None or torso.size == 0:
            return None

        mask = self._mask_non_green(torso)
        if mask.sum() < 30:  # pocos píxeles válidos
            # si no quedan muchos no-verdes, tomar todo el torso como fallback
            pixels = torso.reshape(-1, 3)
        else:
            pixels = torso[mask].reshape(-1, 3)

        if pixels.shape[0] < 20:
            return None

        # usar centroid (media) en lugar de KMeans por jugador — más rápido y estable
        color = pixels.mean(axis=0)  # BGR
        # convertir a RGB para consistencia con centros si usas sklearn (pero mantenemos BGR interno)
        return color.astype(np.float32)

    # ---------------------------
    # Bootstrap (one-shot) de colores de equipo
    # ---------------------------
    def bootstrap_colors(self, frame: MatLike, players: List[PlayerStateModel]) -> bool:
        """
        Entrena MiniBatchKMeans una sola vez cuando haya suficientes colores válidos.
        players: lista de PlayerStateModel con bbox disponible
        """
        if self.kmeans is not None:
            return True

        samples = []
        for player in players:
            bbox = player.get_bbox()
            if not bbox:
                continue
            c = self.extract_player_color(frame, bbox)
            if c is not None:
                samples.append(c)

        if len(samples) < self.min_bootstrap_players:
            logging.debug(f"Bootstrap: need >={self.min_bootstrap_players} valid players, got {len(samples)}")
            return False

        try:
            mbk = MiniBatchKMeans(n_clusters=2, batch_size=32, random_state=0)
            mbk.fit(np.vstack(samples))
            self.kmeans = mbk
            centers = mbk.cluster_centers_
            self.team_colors = {1: centers[0].astype(np.float32), 2: centers[1].astype(np.float32)}
            logging.info("TeamAssigner: bootstrap complete, 2 team colors learned.")
            return True
        except Exception as e:
            logging.debug(f"Bootstrap KMeans failed: {e}")
            return False

    # ---------------------------
    # Predicción rápida por color
    # ---------------------------
    def _predict_from_color(self, color_bgr: np.ndarray) -> Optional[int]:
        if self.kmeans is None:
            return None
        try:
            label = int(self.kmeans.predict(color_bgr.reshape(1, -1))[0])
            return label + 1
        except Exception as e:
            error_logger.error(f"[Team Assigner] KMeans predict error: {e}")
            return None

    # ---------------------------
    # API principal (manteniendo nombre de la clase)
    # ---------------------------
    def assign_team_colors(self, frame: MatLike, players: List[PlayerStateModel]) -> None:
        """
        Método principal: intenta bootstrap si no hay modelo; no reentrena si ya existe.
        """
        if self.kmeans is None:
            self.bootstrap_colors(frame, players)
        # Aquí podrías actualizar cache, smoothing o predicciones por frame

    def get_player_team(self, frame: MatLike, record: PlayerStateModel, db: Session):
        """
        Devuelve 1, 2 o -1. Utiliza smoothing temporal por jugador
        para evitar saltos por recortes malos.
        """
        try:
        # obtener identificador estable: player_id preferible, si no id
            debug_logger.debug("[Team Assigner] Getting player team for record:", record)
            player_record = TrackCollectionPlayer(db)
            player_id = getattr(record, "id", None)
            if player_id is None:
                debug_logger.debug("[Team Assigner] Record has no id attribute.")
                return -1, None

            bbox = record.get_bbox()
            debug_logger.debug(f"[Team Assigner] Player ID: {player_id}, BBox: {bbox}")
            if not bbox:
                # usar majority vote de historial
                debug_logger.debug("[Team Assigner] No bbox available, using history for team assignment.")
                hist = self.player_team_history[player_id]
                if len(hist) == 0:
                    return -1, None
                # devolver la decisión cacheada o mayoría
                return self._majority_vote(hist)

            # si no hay modelo entrenado → -1 (o podrías intentar bootstrap local)
            debug_logger.debug("[Team Assigner] Checking KMeans model...")
            if self.kmeans is None:
                debug_logger.debug("[Team Assigner] KMeans not initialized yet when predicting team.")
                self.assign_team_colors(frame=frame, players=player_record.get_all())
                debug_logger.debug("[Team Assigner] After attempting bootstrap.")

            debug_logger.debug("[Team Assigner] Extracting player color...")
            color = self.extract_player_color(frame, bbox)
            debug_logger.debug(f"[Team Assigner] Extracted color: {color}")
            if color is None:
                # no color -> fallback
                debug_logger.debug("[Team Assigner] No pudo extraer color del jugador, usando historial.")
                hist = self.player_team_history[player_id]
                if len(hist) == 0:
                    return -1, None
                return self._majority_vote(hist)

            debug_logger.debug("[Team Assigner] Prediciendo equipo desde color...")
            pred = self._predict_from_color(color)
            debug_logger.debug(f"[Team Assigner] Predicted team: {pred}")
            if pred is None:
                debug_logger.debug("[Team Assigner] No pudo predecir equipo desde color.")
                return -1, None

            # update smoothing history
            self.player_team_history[player_id].append(pred)

            # si la historia tiene suficiente longitud, usar mayoría; sino usar pred
            debug_logger.debug("[Team Assigner] Usando smoothing history...")
            hist = self.player_team_history[player_id]
            debug_logger.debug(f"[Team Assigner] History: {hist}")
            team = self._majority_vote(hist) if len(hist) >= max(3, self.smoothing_window // 2) else pred
            debug_logger.debug(f"[Team Assigner] Team after smoothing: {team}")

            # actualizar cache y devolver
            debug_logger.debug("[Team Assigner] Actualizando cache...")
            self.player_team_cache[player_id] = int(team)
            player_record.patch(
                player_id,
                {
                    "team": team,
                    "color":  json.dumps(self.team_colors.get(team)) if team in self.team_colors else None
                }
            )
            debug_logger.debug(f"[Team Assigner] Equipo asignado al jugador {player_id}: {team}")
            return int(team)
        except Exception as e:
            error_logger.error(f"[Team Assigner] Error predicting team: {e}")
            return -1, None

    # ---------------------------
    # Utilidades
    # ---------------------------
    def _majority_vote(self, hist: deque) -> int:
        if len(hist) == 0:
            return -1
        counts = {}
        for v in hist:
            counts[v] = counts.get(v, 0) + 1
        # devolver valor con mayor ocurrencia; en empate preferir valor anterior cacheado
        sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        top_label = sorted_counts[0][0]
        return int(top_label)

    def reset(self):
        """Resetea estado aprendido (colores, historial)."""
        self.kmeans = None
        self.team_colors = {}
        self.player_team_history.clear()
        self.player_team_cache.clear()
