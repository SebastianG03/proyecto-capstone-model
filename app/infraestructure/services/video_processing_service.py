import pathlib
import time
from typing import Generator, List, Tuple
from uuid import uuid4
from xmlrpc.client import boolean

import cv2
from cv2.typing import MatLike
import numpy as np

from app.core.config import DEBUG
from app.entities.models.PlayerModels import Player, PlayerState
import app.entities.utils.global_values_store as value_store
from app.logger.logger import debug_logger, info_logger, error_logger


def _open_capture(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    return cap

def is_duplicate_frame(
    prev_frame: MatLike,
    current_frame: MatLike,
    threshold: float = 0.95) -> boolean:
    prev = cv2.resize(prev_frame, (64,64))
    curr = cv2.resize(current_frame, (64,64))

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    prev = prev.astype(np.float32) / 255.0
    curr = curr.astype(np.float32) / 255.0

    similarity = np.mean(prev * curr) / (np.std(prev) * np.std(curr) + 1e-8)
    info_logger.info(f"[VideoProcessing] Similaridad del frame: {similarity}")
    return boolean(similarity > threshold)

def normalize_frame(frame: MatLike, target_size = 640) -> MatLike:
    h, w = frame.shape[:2]
    
    scale = max(target_size / min(h, w), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    info_logger.info(f"[VideoProcessing] Tamano del frame: {resized.shape}")
    return resized

def _read_batch(
    cap, batch_size: int
) -> List[Tuple[MatLike, float]]:
    batch: List[Tuple[MatLike, float]] = []
    prev_frame = None

    for _ in range(batch_size):
        frame_exists, frame = cap.read()
        dt = float(cap.get(cv2.CAP_PROP_POS_MSEC))

        if not frame_exists:
            break
        
        if prev_frame is not None and is_duplicate_frame(prev_frame, frame):
            info_logger.info("[VideoProcessing] Ignorando frame duplicado")
            continue
        
        batch.append((normalize_frame(frame), dt))
    return batch

def check_video(video_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    is_video = cap.isOpened()
    cap.release()
    return is_video
    
def get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    cap.release()
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def read_video(
    video_path: str, batch_size: int = 16
) -> Generator[List[Tuple[MatLike, float]], None, None]:
    
    if batch_size <= 6:
        raise ValueError("El tamano del batch debe ser mayor a 6")
    
    info_logger.info(f"Abriendo video: {video_path}")
    cap = _open_capture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    value_store.globals.frame_size = video_size
    
    info_logger.info(f"[READ VIDEO] Total frames: {total_frames}, FPS: {frame_rate}")
    if frame_rate != value_store.globals.fps:
        info_logger.info(f"FPS detectado: {frame_rate}, actualizando valor global.")
        value_store.globals.update(fps=frame_rate)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last_time = time.time()
        frame_count = 0

        while frame_count < total_frames:
            batch = _read_batch(cap, batch_size)
            last_time = time.time()
            frame_count += len(batch)
            if batch:
                debug_logger.debug(f"Yielding batch of size {len(batch)}")
                yield batch
            else:
                break
    except FileNotFoundError as e:
        error_logger.error(str(e))
        raise RuntimeError(
            "No se pudo abrir el video especificado. Verifica la ruta y los permisos."
        )
    except Exception:
        error_logger.exception("Error inesperado al leer el video")
        raise RuntimeError(
            "Ocurrio un error procesando el video. Revisa los logs para mas detalles."
        )
    finally:
        cap.release()


def _validate_and_normalize_bbox(bbox, w, h):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return x1, y1, x2, y2


def _save_player_crop(
    folder: pathlib.Path,
    crop,
    player_id: int,
    player_team,
    player_color,
    frame_index: int,
):
    filename = (
        folder
        / (
            f"player_{player_id}_team_{player_team}_color_{player_color}_"
            f"img_{uuid4()}_frame_{frame_index}.png"
        )
    )
    cv2.imwrite(str(filename), crop)
    return filename


def extract_player_images(
    frame: MatLike,
    frame_index: int,
    player_state: PlayerState,
    player: Player,
    output_folder: str,
):
    try:
        if not DEBUG:
            return

        if frame_index % 30 != 0:
            return

        debug_logger.debug(f"Extrayendo imagen de jugador en frame {frame_index}")
        folder = pathlib.Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)


        h, w = frame.shape[:2]

        state_record = player_state.to_dict()
        player_record = player.to_dict() if player else {}
        player_id = int(state_record.get("player_id", -1))
        if player_id == -1:
            debug_logger.debug("Player ID invalido, omitiendo.")
            return

        bbox = player_state.get_bbox()
        if not bbox or len(bbox) != 4:
            debug_logger.debug("BBox ausente o invalido.")
            return

        bbox_norm = _validate_and_normalize_bbox(bbox, w, h)
        if bbox_norm is None:
            debug_logger.debug("BBox normalizada invalida.")
            return

        x1, y1, x2, y2 = bbox_norm
        torso_y2 = y1 + int((y2 - y1) * 0.6)
        crop = frame[y1:torso_y2, x1:x2]
        if crop.size == 0:
            debug_logger.debug("Crop vacio, omitiendo.")
            return

        player_team = player_record.get("team", "unknown")
        player_color = player_record.get("color", "unknown")
        filename = _save_player_crop(
            folder, crop, player_id, player_team, player_color, frame_index
        )

        debug_logger.debug(f"Imagen guardada: {filename}")
    except Exception:
        error_logger.exception("Error al extraer la imagen del jugador")
        raise RuntimeError(
            "No se pudo extraer la imagen del jugador. "
            "Por favor revisa el video y los parametros de entrada."
        )
