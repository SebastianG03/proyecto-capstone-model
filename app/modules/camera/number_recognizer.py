
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session
from app.logger import info_logger, error_logger
from app.modules.camera.number_recognition import NumberRecognition

class PlayerNumberRecognitionService:
    def __init__(self, model_path: str):
        self.model = NumberRecognition(model_path)
        self.history: Dict[int, List[int]] = {}  # player_id -> números detectados

    def _validate_consistency(self, player_id: int, number: int) -> Optional[int]:
        if player_id not in self.history:
            self.history[player_id] = []
        self.history[player_id].append(number)
        history = self.history[player_id][-10:]

        counts: Dict[int, int] = {}
        for n in history:
            counts[n] = counts.get(n, 0) + 1

        most_common, freq = max(counts.items(), key=lambda x: x[1])
        if freq >= 3:
            return most_common
        return None

    def update_player_number(
        self,
        db: Session,
        player_id: int,
        frame_index: int,
        back_roi: np.ndarray
    ) -> None:
        try:
            number = self.model.predict(back_roi)
            if number is None:
                return

            validated = self._validate_consistency(player_id, number)
            if validated is None:
                return

            #Update player

        except Exception as e:
            error_logger.error(f"Error al actualizar número de jugador {player_id}: {e}")