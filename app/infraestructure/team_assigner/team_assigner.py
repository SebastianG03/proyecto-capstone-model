import json
import logging
import os
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from cv2.typing import MatLike
from sqlalchemy.orm import Session

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState
from app.entities.utils.singleton import Singleton

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


class TeamAssigner(metaclass=Singleton):
    """
    Versión optimizada para uso frame-a-frame sin renombrar la clase.
    - Bootstrap (one-shot) de colores de equipo.
    - Clasificación O(1) por jugador por frame.
    - Smoothing temporal por jugador (ventana configurable).
    - Toma torso central + K-means RGB simple (3 clusters).
    - Fallbacks robustos si bbox está recortado o inválido.
    """

    def __init__(
        self,
        smoothing_window: int = 11,
        min_bootstrap_players: int = 8,
        torso_fraction: float = 0.4,
    ):
        # estado del modelo
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.last_kmeans_train_length = 0
        self.team_colors: Dict[int, np.ndarray] = {}

        # cache y smoothing
        self.player_team_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )
        self.player_team_cache: Dict[int, int] = {}  # última decisión estable
        self._last_seen: Dict[int, int] = {}

        # parámetros
        self.smoothing_window = smoothing_window
        self.min_bootstrap_players = min_bootstrap_players
        self.torso_fraction = float(np.clip(torso_fraction, 0.2, 0.6))

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
        start = int(h * 0.2)  # ignorar cabeza ~20 %
        end = int(h * (0.2 + self.torso_fraction))
        if end <= start:
            return None
        return crop[start:end, :]

    # ---------------------------
    # Normaliza iluminación
    # ---------------------------
    def _illuminant_normalize(self, bgr: np.ndarray) -> np.ndarray:
        """Gray-world simple: divide por la media de cada canal."""
        bgr = bgr.astype(np.float32)
        mu = bgr.reshape(-1, 3).mean(axis=0)
        gray = mu.mean()
        scale = gray / (mu + 1e-6)
        return np.clip(bgr * scale, 0, 255).astype(np.uint8)

    # --------------------------------------------------
    #  extract_player_color  -> K-means RGB simple
    # --------------------------------------------------
    def extract_player_color(self, frame: MatLike, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Devuelve el color dominante (BGR, float32) del torso del jugador.
        """
        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None
        torso = self._torso_region(crop)
        if torso is None or torso.size == 0:
            return None

        # downsample si imagen grande
        h, w = torso.shape[:2]
        if h * w > 90_000:
            torso = cv2.resize(torso, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        # 1. Iluminancia constante
        torso = self._illuminant_normalize(torso)

        # 2. RGB plano
        rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape(-1, 3)

        # 3. K-means 3 clusters
        km = MiniBatchKMeans(n_clusters=3, batch_size=2048, random_state=0)
        labels = km.fit_predict(pixels)
        centers = km.cluster_centers_

        # 4. ¿cuál es el más poblado?
        uniq, counts = np.unique(labels, return_counts=True)
        best = uniq[np.argmax(counts)]
        dominant_rgb = centers[best]

        # 5. De vuelta a BGR y listo
        dominant_bgr = cv2.cvtColor(
            dominant_rgb.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2BGR
        )[0, 0]

        return dominant_bgr.astype(np.float64)  # 0-255 seguro

    # ---------------------------
    # Bootstrap (one-shot) de colores de equipo
    # ---------------------------
    def bootstrap_colors(self, frame: MatLike, players: List[PlayerState]) -> bool:
        """
        Entrena MiniBatchKMeans una sola vez cuando haya suficientes colores válidos.
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
            logger.debug(f"Bootstrap: need >={self.min_bootstrap_players} valid players, got {len(samples)}")
            return False

        try:
            mbk = MiniBatchKMeans(n_clusters=2, batch_size=32)
            mbk.fit(np.vstack(samples))
            self.kmeans = mbk
            centers = mbk.cluster_centers_
            self.team_colors = {1: centers[0].astype(np.float32), 2: centers[1].astype(np.float32)}
            logger.info("TeamAssigner: bootstrap complete, 2 team colors learned.")
            return True
        except Exception as e:
            logger.debug(f"Bootstrap KMeans failed: {e}")
            return False

    # ---------------------------
    # Predicción rápida por color
    # ---------------------------
    def _to_lab(self, bgr: np.ndarray) -> np.ndarray:
        arr = np.asarray(bgr, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        return lab

    def _closest_team_from_color(self, color_bgr: np.ndarray) -> Optional[int]:
        if not self.team_colors:
            return None
        lab_c = self._to_lab(color_bgr)
        centers = np.array(list(self.team_colors.values()), dtype=np.uint8)
        labs = np.array([self._to_lab(c) for c in centers])
        dif = labs - lab_c
        dist = np.einsum('ij,ij->i', dif, dif)
        return int(np.argmin(dist)) + 1

    def _predict_from_color(self, color_bgr: np.ndarray) -> Optional[int]:
        if self.kmeans is None:
            return self._closest_team_from_color(color_bgr)
        try:
            X = np.atleast_2d(color_bgr.astype(np.float64))
            label = int(self.kmeans.predict(X)[0])
            return label + 1
        except Exception as e:
            logger.error("KMeans predict error: %s", e, exc_info=True)
            return self._closest_team_from_color(color_bgr)
    # ---------------------------
    # API principal
    # ---------------------------
    def assign_team_colors(self, frame: MatLike, players: List[PlayerState]) -> None:
        if self.kmeans is None or (len(players) % self.min_bootstrap_players == 0 and len(players) != self.last_kmeans_train_length):
            self.last_kmeans_train_length = len(players)
            self.bootstrap_colors(frame, players)

    def get_player_team(self, frame: MatLike, record: PlayerState, frame_num: int, db: Session) -> int:
        """
        Devuelve 1, 2 o -1. Utiliza smoothing temporal por jugador.
        """
        try:
            player_id = int(f'{record.player_id}')
            bbox = record.get_bbox()

            # cleanup cada 300 frames
            if frame_num % 300 == 0:
                cutoff = frame_num - self.smoothing_window
                to_del = [pid for pid, f in self._last_seen.items() if f < cutoff]
                for pid in to_del:
                    self.player_team_history.pop(pid, None)
                    self.player_team_cache.pop(pid, None)
                    self._last_seen.pop(pid, None)

            if not bbox:
                hist = self.player_team_history[player_id]
                return self._majority_vote(hist) if hist else -1

            self._last_seen[player_id] = frame_num

            if self.kmeans is None:
                player_record = TrackCollectionPlayer(db)
                self.assign_team_colors(frame=frame, players=player_record.get_all_states())

            color = self.extract_player_color(frame, bbox)
            if color is None:
                hist = self.player_team_history[player_id]
                return self._majority_vote(hist) if hist else -1

            pred = self._predict_from_color(color)
            if pred is None:
                hist = self.player_team_history[player_id]
                return self._majority_vote(hist) if hist else -1

            self.player_team_history[player_id].append(pred)
            hist = self.player_team_history[player_id]
            team = self._majority_vote(hist) if len(hist) >= max(3, self.smoothing_window // 2) else pred
            team = int(team)
            self.player_team_cache[player_id] = team

            # async db write
            player_record = TrackCollectionPlayer(db)
            player_data = player_record.get_player(int(f'{record.player_id}'))
            if player_data:
                _executor.submit(self._async_patch, player_record, player_data.id, team,
                                 self.team_colors.get(team))
            return team
        except Exception as e:
            logger.error("Error predicting team: %s", e, exc_info=True)
            return -1


    def _async_patch(self, player_record, db_id, team, color):
        try:
            color_str = json.dumps(color.tolist()) if color is not None and color.any() else None
            player_record.patch(db_id, {"team": team, "color": color_str})
        except Exception as e:
            logger.error("Async patch failed: %s", e, exc_info=True)

    def _majority_vote(self, hist: deque) -> int:
        if not hist:
            return -1
        values = np.array(hist)
        uniq, counts = np.unique(values, return_counts=True)
        return int(uniq[np.argmax(counts)])

    def reset(self):
        """Resetea estado aprendido (colores, historial)."""
        self.kmeans = None
        self.team_colors.clear()
        self.player_team_history.clear()
        self.player_team_cache.clear()
        self._last_seen.clear()
