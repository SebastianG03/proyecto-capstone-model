from typing import Tuple
import numpy as np


def get_center_of_bbox(bbox) -> tuple[int,int]:
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox) -> int:
    return bbox[2] - bbox[0]

def measure_scalar_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos.
    
    Args:
        p1, p2: Arrays de numpy con coordenadas [x, y]
        
    Returns:
        Distancia euclidiana como float
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    diff = p1 - p2
    return np.linalg.norm(diff)

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
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int(y2)

def rectangle_coords(width: int, height: int, center: int, y2: int) -> tuple[int, int, int, int]:
    x1_rect = center - width // 2
    x2_rect = center + width // 2
    y1_rect = (y2- height // 2) + 15
    y2_rect = (y2+ height // 2) + 15
    return x1_rect, x2_rect, y1_rect, y2_rect