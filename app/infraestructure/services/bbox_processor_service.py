from typing import Tuple

import cv2
import numpy as np
from scipy.spatial import distance as dist
from cv2.typing import MatLike
from app.logger import debug_logger, error_logger


def get_center_of_bbox(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox) -> int:
    return bbox[2] - bbox[0]


def measure_scalar_distance(p1, p2) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos.

    Args:
        p1, p2: Arrays de numpy con coordenadas [x, y]

    Returns:
        Distancia euclidiana como float
    """
    distance = dist.euclidean(p1, p2)
    debug_logger.debug(f"[MeasureScalarDistance] Distancia euclidiana: {distance}")
    return distance


def measure_vectorial_distance(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    """
    Calcula la diferencia vectorial entre dos puntos.

    Args:
        p1, p2: Arrays de numpy con coordenadas [x, y]

    Returns:
        Tupla con las diferencias (dx, dy)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    diff = p1 - p2
    return float(diff[0]), float(diff[1])


def get_foot_position(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def rectangle_coords(
    width: int, height: int, center: int, y2: int
) -> tuple[int, int, int, int]:
    x1_rect = center - width // 2
    x2_rect = center + width // 2
    y1_rect = (y2 - height // 2) + 15
    y2_rect = (y2 + height // 2) + 15
    return x1_rect, x2_rect, y1_rect, y2_rect


def calculate_area_boundary_ends(
    frame: MatLike,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Recibe un frame TOP-VIEW (MatLike) y devuelve los extremos de la línea
    de área (11 m) en coordenadas del mismo espacio que usas para jugadores.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        debug_logger.debug(
            "[CalculateAreaBoundaryEnds] Detectando líneas en el frame para encontrar la línea de área."
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=int(0.50 * frame.shape[1]),
            maxLineGap=15,
        )
        if lines is None:
            debug_logger.debug(
                "[CalculateAreaBoundaryEnds] No se detectaron líneas en el frame."
            )
            return None

        # quedarse solo con horizontales
        hor = [
            line[0]
            for line in lines
            if abs(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0])) < 0.09
        ]  # ~5°
        if not hor:
            debug_logger.debug(
                "[CalculateAreaBoundaryEnds] No se detectaron líneas horizontales en el frame."
            )
            return None

        # la más baja (línea de área inferior)
        x1, y1, x2, y2 = max(hor, key=lambda line: max(line[1], line[3]))
        debug_logger.debug(
            "[CalculateAreaBoundaryEnds] Línea de área detectada en coordenadas: "
            f"({x1}, {y1}), ({x2}, {y2})"
        )
        A = np.array([min(x1, x2), float(y1)], dtype=float)
        B = np.array([max(x1, x2), float(y2)], dtype=float)
        debug_logger.debug(
            "[CalculateAreaBoundaryEnds] Extremos de la línea de área: "
            f"A={A}, B={B}"
        )
        return A, B
    except Exception as e:
        error_logger.error(f"[CalculateAreaBoundaryEnds] Error: {e}")
        return None


def calculate_meters_per_pixel(
    p1: np.ndarray, p2: np.ndarray, real_distance_m: float
) -> float:
    """
    Calcula la cantidad de metros por píxel dada una distancia real y dos puntos.
    """
    try:
        p1 = np.array(p1)
        p2 = np.array(p2)
        pixel_distance = dist.euclidean(p1, p2)
        debug_logger.debug(
            f"[CalculateMetersPerPixel] Distancia en píxeles entre puntos: {pixel_distance}"
        )
        return real_distance_m / pixel_distance
    except Exception as e:
        error_logger.error(
            f"[CalculateMetersPerPixel] Error calculando metros por píxel: {e}"
        )
        return 0.0
