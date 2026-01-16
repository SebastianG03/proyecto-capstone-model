import json
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from cv2.typing import MatLike
from sqlalchemy.orm import Session

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from app.entities.collections import TrackCollectionPlayer
from app.entities.models.PlayerModels import PlayerState, Player
from app.entities.utils.singleton import Singleton
from app.logger import *

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
    #  NUEVO: extract_player_color  -> K-means RGB simple
    # --------------------------------------------------
    def extract_player_color(self, frame: MatLike, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Devuelve el color dominante (BGR, float32) del torso del jugador.
        Copia exacta del pipeline que ya te funcionaba:
        - crop + torso
        - normaliza iluminación
        - K-means 3 clusters sobre RGB
        - se queda con el cluster más grande
        """
        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None
        torso = self._torso_region(crop)
        if torso is None or torso.size == 0:
            return None

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

        return dominant_bgr.astype(np.float32)  # 0-255 seguro

    # ---------------------------
    # Bootstrap (one-shot) de colores de equipo
    # ---------------------------
    def bootstrap_colors(self, frame: MatLike, players: List[PlayerState]) -> bool:
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
    def _lab_distance(self, lab1: np.ndarray, lab2: np.ndarray, wL: float = 0.2) -> float:
        dl = float(lab1[0] - lab2[0])
        da = float(lab1[1] - lab2[1])
        db = float(lab1[2] - lab2[2])
        return (wL * (dl ** 2) + da ** 2 + db ** 2) ** 0.5

    def _to_lab(self, bgr: np.ndarray) -> np.ndarray:
        # Convert BGR 0-255 (shape (3,) or (1,1,3)) to Lab float32
        arr = np.asarray(bgr, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        return lab

    def _closest_team_from_color(self, color_bgr: np.ndarray) -> Optional[int]:
        if not self.team_colors:
            return None
        try:
            lab_c = self._to_lab(color_bgr)
            best = None
            best_d = float('inf')
            for team, col in self.team_colors.items():
                lab_t = self._to_lab(np.array(col, dtype=np.uint8))
                d = self._lab_distance(lab_c, lab_t)
                if d < best_d:
                    best_d = d
                    best = team
            return best
        except Exception as e:
            error_logger.error(f"[Team Assigner] Closest color error: {e}")
            return None

    def _predict_from_color(self, color_bgr: np.ndarray) -> Optional[int]:
        # Prefer KMeans prediction; on failure fallback to Lab-distance nearest
        if self.kmeans is None:
            return self._closest_team_from_color(color_bgr)
        try:
            label = int(self.kmeans.predict(color_bgr.reshape(1, -1))[0])
            return label + 1
        except Exception as e:
            error_logger.error(f"[Team Assigner] KMeans predict error: {e}, falling back to Lab distance")
            return self._closest_team_from_color(color_bgr)


    # ---------------------------
    # API principal (manteniendo nombre de la clase)
    # ---------------------------
    def assign_team_colors(self, frame: MatLike, players: List[PlayerState]) -> None:
        """
        Método principal: intenta bootstrap si no hay modelo; no reentrena si ya existe.
        """
        if self.kmeans is None or (len(players) % self.min_bootstrap_players == 0 and len(players) != self.last_kmeans_train_length):
            self.last_kmeans_train_length = len(players)
            debug_logger.debug(f"[Team Assigner] Actual train length: {self.last_kmeans_train_length}")
            self.bootstrap_colors(frame, players)
        # Aquí podrías actualizar cache, smoothing o predicciones por frame

    def get_player_team(self, frame: MatLike, record: PlayerState, frame_num: int, db: Session):
        """
        Devuelve 1, 2 o -1. Utiliza smoothing temporal por jugador
        para evitar saltos por recortes malos.
        """
        try:
            # obtener identificador estable: player_id preferible, si no id
            debug_logger.debug(f"[Team Assigner] Getting player team for record: {record}")
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
                self.assign_team_colors(frame=frame, players=player_record.get_all_states())
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
            debug_logger.debug(f"[Team Assigner] Team selected: {team} y color selected: {self.team_colors.get(team)}")
            color = self.team_colors.get(team) if team in self.team_colors else None
            player_data = player_record.get_player(int(f'{record.player_id}'))
            if not player_data:
                raise ValueError(f"[Team Assigner] No se obtuvo el jugador con player id {record.player_id}")
            
            debug_logger.debug(f"[Team Assigner] Datos del jugador obtenido: {player_data.to_dict()}")
            player_record.patch(
                int(f'{player_data.id}'),
                {
                    "team": team,
                    "color": json.dumps(color.tolist()) if color is not None and color.any() else None
                }
            )
            debug_logger.debug(f"[Team Assigner] Equipo asignado al jugador {player_id}, con id {player_data.id}: {team}")
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
