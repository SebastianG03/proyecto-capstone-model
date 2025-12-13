import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.entities.collections import TrackCollectionBall, TrackCollectionPlayer
from app.logger import debug_logger, error_logger, info_logger

class ViewTransformer:
    """
    Transforma un punto desde coordenadas de imagen a coordenadas reales del campo.
    Se integra con RecordCollectionBase para persistir puntos transformados.
    """

    def __init__(
        self,
        field_length_m: float = 105.0,
        field_width_m: float = 68.0):
        # Medidas reales del campo de fútbol (en metros)
        FIELD_LENGTH = field_length_m
        FIELD_WIDTH = field_width_m

        # Coordenadas del polígono detectado en la imagen (4 puntos en píxeles)
        self.pixel_vertices = np.array([
            [110, 1035],   # Bottom-left
            [265, 275],    # Top-left
            [910, 260],    # Top-right
            [1640, 915]    # Bottom-right
        ], dtype=np.float32)

        # Corregir contorno para pointPolygonTest
        self.pixel_vertices_contour = self.pixel_vertices.reshape((-1, 1, 2))

        # Mapa de proyección destino en metros
        self.target_vertices = np.array([
            [0, FIELD_WIDTH],     # Bottom-left
            [0, 0],               # Top-left
            [FIELD_LENGTH, 0],    # Top-right
            [FIELD_LENGTH, FIELD_WIDTH]  # Bottom-right
        ], dtype=np.float32)

        # Matriz de transformación perspectiva
        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices
        )

    # ---------------------------------------------------------
    # TRANSFORMACIÓN DE UN PUNTO INDIVIDUAL
    # ---------------------------------------------------------

    def transform_point(self, point_xy: np.ndarray):
        """
        Transforma un punto x,y a coordenadas reales del campo.
        Retorna None si el punto está fuera del polígono del campo.
        """
        x, y = float(point_xy[0]), float(point_xy[1])
        point_int = (int(x), int(y))

        # Validación: fuera del campo
        if cv2.pointPolygonTest(self.pixel_vertices_contour, point_int, False) < 0:
            return None

        # OpenCV requiere shape (1,1,2)
        p = np.array([[[x, y]]], dtype=np.float32)

        warped = cv2.perspectiveTransform(p, self.perspective_transform)

        return warped.reshape(2).tolist()  # [x,y] en metros

    # ---------------------------------------------------------
    # INTEGRACIÓN CON RECORD COLLECTION BASE
    # ---------------------------------------------------------

    def add_transformed_positions(self, db: Session):
        """
        Ajusta la posición del balón y los jugadores en cada frame.
        Luego de una base de datos, transforma los registros de balón y jugadores.
        Si hay registros de balón pero no de jugadores, solo transforma los registros de balón.
        Si hay registros de jugadores pero no de balón, solo transforma los registros de jugadores.
        Si no hay registros de balón ni de jugadores, no hace nada.
        """
        ball_collection = TrackCollectionBall(db)
        player_collection = TrackCollectionPlayer(db)

        info_logger.info("Transformando posiciones en registros...")
        try:
            ball_records = ball_collection.get_all()
            player_records = player_collection.get_all()
            if len(ball_records) == 0 and len(player_records) == 0:
                info_logger.info("No hay registros para transformar.")
                return
            
            if len(ball_records) == len(player_records):
                self.transform_both_records(ball_records, player_records, db)
                return    
            if len(ball_records) > 0:
                self.transform_ball_records(ball_records, db)
                return
            if len(player_records) > 0:
                self.transform_player_records(player_records, db)
                return
            info_logger.info("No hay registros para transformar.")
        except Exception as e:
            error_logger.error(f"Error transformando posiciones en records: {e}")
            raise e

    def transform_ball_records(self, ball_records, db: Session):
        """
        Transforma la posición del balón en cada frame.
        Luego de una lista de objetos BallEventModel y una sesión de base de datos,
        transforma la posición del balón en cada frame.
        Si hay un error al momento de transformar, se imprime el error y se
        lanza una excepción.
        """
        try:
            for br in ball_records:
                self.calculate_ball_transformed_position(br, db)
        except Exception as e:
            error_logger.error(f"Error transformando posiciones en ball records: {e}")
            raise e

    def transform_player_records(self, player_records, db: Session):
        """
        Transforma la posición de los jugadores en cada frame.
        Luego de una lista de objetos PlayerStateModel y una sesión de base de datos,
        transforma la posición de los jugadores en cada frame.
        Si hay un error al momento de transformar, se imprime el error y se
        lanza una excepción.
        """
        try:
            for pr in player_records:
                self.calculate_player_transformed_position(pr, db)
        except Exception as e:
            error_logger.error(f"Error transformando posiciones en player records: {e}")
            raise e

    def transform_both_records(self, ball_records, player_records, db: Session):
        """
        Transforma la posición del balón y los jugadores en cada frame.
        Luego de una lista de objetos BallEventModel y una lista de objetos PlayerStateModel
        y una sesión de base de datos, transforma la posición del balón y los
        jugadores en cada frame.
        Si hay un error al momento de transformar, se imprime el error y se
        lanza una excepción.
        """
        try:
            for br, pr in zip(ball_records, player_records):
                self.calculate_ball_transformed_position(br, db)
                self.calculate_player_transformed_position(pr, db)
        except Exception as e:
            error_logger.error(f"Error transformando posiciones en records: {e}")
            raise e
        
    def calculate_ball_transformed_position(self, ball_record, db: Session):
        """
        Calcula la posición transformada del balón en un registro.
        
        Recibe un registro BallEventModel y una sesión de base de datos.
        Transforma la posición del balón en el registro y la guarda en la base
        de datos.
        
        Si hay un error al momento de transformar, se imprime el error y se
        lanza una excepción.
        """
        try:
            bx, by = ball_record.x, ball_record.y
            if bx is None and by is None:
                return
            ball_transformed = self.transform_point(np.array([bx, by], dtype=np.float32))
            if ball_transformed is None:
                return
            ball_collection = TrackCollectionBall(db)
            ball_collection.patch(
                ball_record.id,
                {"x_transformed": ball_transformed[0],
                "y_transformed": ball_transformed[1]}
            )
            return
        except Exception as e:
            error_logger.error(f"Error calculando posición transformada del balón: {e}")
            raise e

    def calculate_player_transformed_position(self, player_record, db: Session):
        """
        Calcula la posición transformada del jugador en un registro.
        
        Recibe un registro PlayerStateModel y una sesión de base de datos.
        Transforma la posición del usuario en el registro y la guarda en la base
        de datos.
        
        Si hay un error al momento de transformar, se imprime el error y se
        lanza una excepción.
        """
        try:
            px, py = player_record.x, player_record.y
            if px is None and py is None:
                return
            player_transformed = self.transform_point(np.array([px, py], dtype=np.float32))
            if player_transformed is None:
                return
            player_collection = TrackCollectionPlayer(db)
            player_collection.patch(
                player_record.id,
                {"x_transformed": player_transformed[0],
                "y_transformed": player_transformed[1]}
            )
            return
        except Exception as e:
            error_logger.error(f"Error calculando posición transformada del jugador: {e}")
            raise e
