from app.entities.utils.singleton import Singleton


class PlayerNumberRecognizer(metaclass=Singleton):
    def __init__(self):
        pass

    def recognize_player_number(self, player_image) -> int:
        """
        Reconoce el número del jugador a partir de su imagen.

        Parámetros
        ----------
        player_image : np.ndarray
            Imagen del jugador recortada.

        Retorna
        -------
        int
            Número reconocido del jugador. Retorna -1 si no se reconoce ningún número.
        """
        # Implementación ficticia para ilustrar
        recognized_number = -1  # Lógica de reconocimiento aquí
        return recognized_number