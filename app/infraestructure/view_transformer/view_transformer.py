import math
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.entities.collections import TrackCollectionBall, TrackCollectionPlayer
from app.entities.models.BallState import BallEventModel
from app.entities.models.PlayerModels import PlayerState, Player
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

        info_logger.info("[ViewTransformer] Transformando posiciones en registros...")
        try:
            ball_register = ball_collection.get_last()
            player_register = player_collection.get_last()
            info_logger.info(f"[ViewTransformer] Registro de balon extraido: {ball_register is not None}")
            info_logger.info(f"[ViewTransformer] Registro de jugador extraido: {player_register is not None}")
            if ball_register is not None:
                debug_logger.debug(f"[ViewTransformer] Calculando posición transformada del balon, datos del balon {ball_register.to_dict()}")
                self.calculate_ball_transformed_position(ball_record=ball_register, db=db)
            if player_register is not None:
                debug_logger.debug(f"[ViewTransformer] Calculando posición transformada del jugador, datos del jugador {player_register.to_dict()}")
                self.calculate_player_transformed_position(player_record=player_register, db=db)

            info_logger.info("[ViewTransformer] No hay registros para transformar.")
        except Exception as e:
            error_logger.error(f"[ViewTransformer] Error transformando posiciones en records: {e}")
            raise e


    def calculate_ball_transformed_position(
        self,
        ball_record: BallEventModel,
        db: Session):
        """
        Calcula la posición transformada del balón en un registro.
        
        Recibe un registro BallEventModel y una sesión de base de datos.
        Transforma la posición del balón en el registro y la guarda en la base
        de datos.
        
        Si hay un error al momento de transformar, se imprime el error y se
        lanza una excepción.
        """
        try:
            bx, by = self._validate_player_coordinates(f'{ball_record.x}', f'{ball_record.y}')
            debug_logger.debug(f"[ViewTransformer] Posición del balón para transformación: x={bx}, y={by}")
            if bx is None and by is None:
                debug_logger.debug("[ViewTransformer] No hay posiciones para transformar.")
                return
            ball_transformed = self.transform_point(np.array([bx, by], dtype=np.float32))
            debug_logger.debug(f"[ViewTransformer] Posición transformada del balón: {ball_transformed}")
            if ball_transformed is None:
                debug_logger.debug("[ViewTransformer] Posición transformada es None, fuera del campo.")
                return
            ball_collection = TrackCollectionBall(db)
            ball_collection.patch(
                int(f'{ball_record.id}'),
                {"x_transformed": ball_transformed[0],
                "y_transformed": ball_transformed[1]}
            )
            return
        except Exception as e:
            error_logger.error(f"[ViewTransformer] Error calculando posición transformada del balón: {e}")
            raise e
        
    def _validate_player_coordinates(self, x, y):
        """Valida coordenadas de jugadores con protección extra contra valores corruptos"""
        try:
            # Convertir a float con manejo de valores extremos
            px = float(x) if x is not None else None
            py = float(y) if y is not None else None
            
            if px is None or py is None:
                return None, None
                
            # Verificar si son infinitos o NaN
            if (math.isinf(px) or math.isnan(px) or 
                math.isinf(py) or math.isnan(py)):
                error_logger.error(f"[ViewTransformer] Coordenadas inválidas detectadas: x={px}, y={py}")
                return None, None
                
            max_reasonable_value = 99999999999
            if abs(px) > max_reasonable_value or abs(py) > max_reasonable_value:
                error_logger.error(f"[ViewTransformer] Coordenadas corruptas (demasiado grandes): x={px}, y={py}")
                return None, None
                
            return px, py
            
        except (ValueError, TypeError) as e:
            error_logger.error(f"[ViewTransformer] Error al validar coordenadas de jugador: {e}")
            return None, None

    def calculate_player_transformed_position(
        self,
        player_record: PlayerState,
        db: Session):
        """
        Calcula la posición transformada del jugador en un registro.
        
        Recibe un registro PlayerStateModel y una sesión de base de datos.
        Transforma la posición del usuario en el registro y la guarda en la base
        de datos.
        
        Si hay un error al momento de transformar, se imprime el error y se
        lanza una excepción.
        """
        try:
            px, py = float(f'{player_record.x}'), float(f'{player_record.y}')
            debug_logger.debug(f"[ViewTransformer] Posición del jugador para transformación: x={px}, y={py}")
            if px is None and py is None:
                debug_logger.debug("[ViewTransformer] No hay posiciones para transformar.")
                return
            player_transformed = self.transform_point(np.array([px, py], dtype=np.float32))
            debug_logger.debug(f"[ViewTransformer] Posición transformada del jugador: {player_transformed}")
            if player_transformed is None:
                debug_logger.debug("[ViewTransformer] Posición transformada es None, fuera del campo.")
                return
            player_collection = TrackCollectionPlayer(db)
            player_collection.patch_state(
                int(f'{player_record.player_id}'),
                int(f'{player_record.frame_index}'),
                {"x_transformed": player_transformed[0],
                "y_transformed": player_transformed[1]}
            )
            return
        except Exception as e:
            error_logger.error(f"[ViewTransformer] Error calculando posición transformada del jugador: {e}")
            raise e
