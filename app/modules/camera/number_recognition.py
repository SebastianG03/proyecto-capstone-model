import cv2
import numpy as np
from typing import Optional
from keras.models import load_model

class NumberRecognition:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.input_size = (64, 64)
        self.min_confidence = 0.85

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        resized = cv2.resize(clean, self.input_size)
        normalized = resized.astype('float32') / 255.0
        return np.expand_dims(normalized, axis=-1)  # (64, 64, 1)

    def predict(self, roi: np.ndarray) -> Optional[int]:
        if self.model is None:
            return None

        processed = self.preprocess(roi)
        pred = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]
        class_id = int(np.argmax(pred))  # 🔧 Conversión explícita a int
        confidence = float(pred[class_id])

        if confidence >= self.min_confidence:
            return class_id + 1  # 0-98 → 1-99
        return None