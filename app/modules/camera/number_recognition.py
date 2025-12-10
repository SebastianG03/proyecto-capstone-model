import cv2
import numpy as np
from tensorflow.keras.models import load_model
from typing import List, Tuple, Optional

class NumberRecognition:
    def __init__(self, model_path: str | None = None):
        """
        Inicializa el reconocedor de números
        
        Args:
            model_path: Ruta al modelo pre-entrenado
        """
        self.model = None
        self.min_confidence = 0.85
        self.number_roi_size = (64, 64)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Carga el modelo de reconocimiento de números"""
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            # Usar OCR como fallback
            self.initialize_ocr_fallback()
    
    def initialize_ocr_fallback(self):
        """Inicializa OCR como método de respaldo"""
        import pytesseract
        self.ocr_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    
    def preprocess_number_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocesa la región de interés del número
        
        Args:
            roi: Región de interés que contiene el número
            
        Returns:
            Imagen preprocesada lista para el modelo
        """
        # Convertir a escala de grises
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Aplicar umbral adaptativo para mejorar contraste
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Eliminar ruido
        kernel = np.ones((2,2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Redimensionar al tamaño esperado por el modelo
        resized = cv2.resize(clean, self.number_roi_size)
        
        # Normalizar
        normalized = resized.astype('float32') / 255.0
        
        # Añadir dimensión para el modelo
        if self.model:
            return np.expand_dims(normalized, axis=-1)
        else:
            return clean
    
    def detect_number_in_back(self, player_back_roi: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Detecta el número en la espalda del jugador
        
        Args:
            player_back_roi: Región de la espalda del jugador
            
        Returns:
            Tupla (número detectado, confianza)
        """
        try:
            # Preprocesar la imagen
            processed = self.preprocess_number_roi(player_back_roi)
            
            if self.model:
                # Usar modelo de deep learning
                prediction = self.model.predict(np.expand_dims(processed, axis=0))
                number = np.argmax(prediction[0])
                confidence = float(prediction[0][number])
                
                # Validar que sea un número válido (1-99)
                if 1 <= number <= 99 and confidence >= self.min_confidence:
                    return number, confidence
            else:
                # Usar OCR como fallback
                return self._ocr_fallback(player_back_roi)
                
        except Exception as e:
            print(f"Error en detección de número: {e}")
        
        return None, 0.0
    
    def _ocr_fallback(self, roi: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Método de respaldo usando OCR
        
        Args:
            roi: Región de interés
            
        Returns:
            Tupla (número detectado, confianza)
        """
        try:
            import pytesseract
            text = pytesseract.image_to_string(roi, config=self.ocr_config)
            text = text.strip()
            
            if text.isdigit():
                number = int(text)
                if 1 <= number <= 99:
                    return number, 0.7  # Confianza media para OCR
            
        except ImportError:
            print("PyTesseract no está instalado")
        
        return None, 0.0
    
    def validate_number_consistency(self, numbers_history: List[int]) -> Optional[int]:
        """
        Valida la consistencia del número detectado en múltiples frames
        
        Args:
            numbers_history: Historial de números detectados
            
        Returns:
            Número más probable o None
        """
        if not numbers_history:
            return None
        
        # Contar frecuencia de cada número
        number_counts = {}
        for num in numbers_history:
            if num is not None:
                number_counts[num] = number_counts.get(num, 0) + 1
        
        if not number_counts:
            return None
        
        # Encontrar el número más frecuente
        most_frequent = max(number_counts.items(), key=lambda x: x[1])
        
        # Requerir al menos 3 detecciones o 60% de consistencia
        total_detections = sum(number_counts.values())
        if most_frequent[1] >= 3 or most_frequent[1] / total_detections >= 0.6:
            return most_frequent[0]
        
        return None