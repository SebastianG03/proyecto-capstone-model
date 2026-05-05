import math
from typing import List, Tuple

import joblib
import numpy as np

from app.entities.models.PlayerModels import PlayerState, State, PlayerStatus
import app.entities.utils.global_values_store as value_store
from app.entities.trackers.player.player_kalman_filter import PlayerKalmanFilter
import app.entities.utils.tools_context as context
from app.logger import info_logger
import xgboost as xgb

from app.utils.routes import PLAYER_XGB_MODEL

class PlayerMatcher():
    def __init__(self,):
        self._training_buffer: list[tuple[np.ndarray, int]] = []
        self.BUFFER_MIN_SAMPLES = 500
        self.NEGATIVE_SAMPLES_PER_MATCH = 2

        self.player_model_exists = (PLAYER_XGB_MODEL / "player_model.pkl").exists()

        if self.player_model_exists:
            self.ml_model = joblib.load((PLAYER_XGB_MODEL / "player_model.pkl").as_posix())
            self.ml_model.set_params(n_estimators=200, max_depth=6, learning_rate=0.05)
        else:
            self.ml_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                objective="binary:logistic"
            )
    
    def _ml_score(self, features: np.ndarray):
        if not self.player_model_exists:
            return 0.5
        try:
            prob = self.ml_model.predict_proba(features.reshape(1, -1))[0][1]
            return float(prob)
        except Exception as e:
            info_logger.warning(f"[PlayerMatcher] ML predict_proba falló: {e}")
            return 0.5

    def fit_ml_model(self, X: np.ndarray, y: np.ndarray):
        """Entrena el modelo con datos acumulados. Llama esto offline."""
        self.ml_model.fit(X, y)
        self._ml_ready = True
        self.save_ml_model()
        info_logger.info(f"[PlayerMatcher] Modelo entrenado con {len(X)} muestras.")
    
    def save_ml_model(self):
        joblib.dump(self.ml_model, PLAYER_XGB_MODEL.as_posix())

    def train_from_buffer(self) -> bool:
        total = len(self._training_buffer)
        if total < self.BUFFER_MIN_SAMPLES:
            info_logger.info(
                f"[PlayerMatcher] Buffer insuficiente para entrenar: "
                f"{total}/{self.BUFFER_MIN_SAMPLES} muestras"
            )
            return False

        X = np.array([f for f, _ in self._training_buffer], dtype=np.float32)
        y = np.array([l for _, l in self._training_buffer], dtype=np.int32)

        pos = int(y.sum())
        neg = int((y == 0).sum())
        info_logger.info(f"[PlayerMatcher] Entrenando con {pos} positivos y {neg} negativos")

        self.fit_ml_model(X, y)
        self._training_buffer.clear()
        return True