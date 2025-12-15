from typing import Tuple

import cv2
import numpy as np
from scipy.spatial import distance as dist
from cv2.typing import MatLike
from app.logger import debug_logger


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
    manual_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    debug_logger.debug(f"[MeasureScalarDistance] Distancia euclidiana (manual): {manual_dist}")
    return distance


def measure_vectorial_distance(
        p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
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
    width: int, height: int, center: int,
    y2: int) -> tuple[int, int, int, int]:
    x1_rect = center - width // 2
    x2_rect = center + width // 2
    y1_rect = (y2 - height // 2) + 15
    y2_rect = (y2 + height // 2) + 15
    return x1_rect, x2_rect, y1_rect, y2_rect

def calculate_area_boundary_ends(frame: MatLike) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Recibe un frame TOP-VIEW (MatLike) y devuelve los extremos de la línea
    de área (11 m) en coordenadas del mismo espacio que usas para jugadores.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    debug_logger.debug("[CalculateAreaBoundaryEnds] Detectando líneas en el frame para encontrar la línea de área.")

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=60,
        minLineLength=int(0.50 * frame.shape[1]), 
        maxLineGap=15
    )
    if lines is None:
        debug_logger.debug("[CalculateAreaBoundaryEnds] No se detectaron líneas en el frame.")
        return None 

    # quedarse solo con horizontales
    hor = [l[0] for l in lines if abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) < 0.09]  # ~5°
    if not hor:
        debug_logger.debug("[CalculateAreaBoundaryEnds] No se detectaron líneas horizontales en el frame.")
        return None

    # la más baja (línea de área inferior)
    x1,y1,x2,y2 = max(hor, key=lambda l: max(l[1], l[3]))
    debug_logger.debug(f"[CalculateAreaBoundaryEnds] Línea de área detectada en coordenadas: ({x1}, {y1}), ({x2}, {y2})")
    A = np.array([min(x1,x2), float(y1)], dtype=float)
    B = np.array([max(x1,x2), float(y2)], dtype=float)
    debug_logger.debug(f"[CalculateAreaBoundaryEnds] Extremos de la línea de área: A={A}, B={B}")
    return A, B
