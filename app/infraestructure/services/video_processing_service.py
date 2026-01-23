import pathlib
import time
from typing import Generator, List, Tuple, Optional

import cv2
from cv2.typing import MatLike

from app.core.config import DEBUG
from app.entities.models.PlayerModels import Player, PlayerState
from app.entities.utils.global_values_store import GlobalValuesStore
from app.logger.logger import debug_logger, info_logger, error_logger

def _open_capture(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    return cap


def _read_batch(cap, batch_size: int, last_time: float) -> Tuple[List[Tuple[MatLike, float]], float]:
    batch = []
    now = time.time()
    dt = now - last_time
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        batch.append((frame, dt))
    return batch, now


def read_video(video_path: str, batch_size: int = 16) -> Generator[List[Tuple[MatLike, float]], None, None]:
    info_logger.info(f"Abriendo video: {video_path}")
    cap = _open_capture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    info_logger.info(f"[READ VIDEO] Total frames: {total_frames}, FPS: {frame_rate}")
    globals = GlobalValuesStore()
    if frame_rate and frame_rate != globals.fps:
        info_logger.info(f"FPS detectado: {frame_rate}, actualizando valor global.")
        globals.update(fps=frame_rate)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last_time = time.time()
        frame_count = 0

        while frame_count < total_frames:
            batch, last_time = _read_batch(cap, batch_size, last_time)
            frame_count += len(batch)
            if batch:
                debug_logger.debug(f"Yielding batch of size {len(batch)}")
                yield batch
            else:
                break
    except FileNotFoundError as e:
        error_logger.error(str(e))
        raise RuntimeError("No se pudo abrir el video especificado. Verifica la ruta y los permisos.")
    except Exception as e:
        error_logger.exception("Error inesperado al leer el video")
        raise RuntimeError("Ocurrió un error procesando el video. Revisa los logs para más detalles.")
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


def _save_player_crop(folder: pathlib.Path, crop, player_id: int, player_team, player_color, count, frame_index: int):
    filename = folder / f"player_{player_id}_team_{player_team}_color_{player_color}_img_{count+1}_frame_{frame_index}.png"
    cv2.imwrite(str(filename), crop)
    return filename


def extract_player_images(
    frame: MatLike,
    frame_index: int,
    player_state: PlayerState,
    player: Player,
    output_folder: str,
    player_image_counts: Optional[dict],
    last_frame_taken: Optional[dict],
    images_per_player: int = 3,
    frame_skip: int = 5,
):
    try:
        if not DEBUG:
            return player_image_counts, last_frame_taken, None
        
        if frame_index % 10 != 0:
            return player_image_counts, last_frame_taken, None
        debug_logger.debug(f"Extrayendo imagen de jugador en frame {frame_index}")
        folder = pathlib.Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)

        player_image_counts = player_image_counts or {}
        last_frame_taken = last_frame_taken or {}

        h, w = frame.shape[:2]

        state_record = player_state.to_dict()
        player_record = player.to_dict() if player else {}
        player_id = int(state_record.get("player_id", -1))
        if player_id == -1:
            debug_logger.debug("Player ID inválido, omitiendo.")
            return player_image_counts, last_frame_taken, None

        bbox = player_state.get_bbox()
        if not bbox or len(bbox) != 4:
            debug_logger.debug("BBox ausente o inválido.")
            return player_image_counts, last_frame_taken, None

        count = player_image_counts.get(player_id, 0)
        if count >= images_per_player:
            return player_image_counts, last_frame_taken, None

        last_f = last_frame_taken.get(player_id, -frame_skip - 1)
        if frame_index - last_f < frame_skip:
            return player_image_counts, last_frame_taken, None

        bbox_norm = _validate_and_normalize_bbox(bbox, w, h)
        if bbox_norm is None:
            debug_logger.debug("BBox normalizada inválida.")
            return player_image_counts, last_frame_taken, None

        x1, y1, x2, y2 = bbox_norm
        torso_y2 = y1 + int((y2 - y1) * 0.6)
        crop = frame[y1:torso_y2, x1:x2]
        if crop.size == 0:
            debug_logger.debug("Crop vacío, omitiendo.")
            return player_image_counts, last_frame_taken, None

        player_team = player_record.get("team", "unknown")
        player_color = player_record.get("color", "unknown")
        filename = _save_player_crop(folder, crop, player_id, player_team, player_color, count, frame_index)

        player_image_counts[player_id] = count + 1
        last_frame_taken[player_id] = frame_index

        debug_logger.debug(f"Imagen guardada: {filename}")
        return player_image_counts, last_frame_taken, player_id
    except Exception:
        error_logger.exception("Error al extraer la imagen del jugador")
        raise RuntimeError("No se pudo extraer la imagen del jugador. Por favor revisa el video y los parámetros de entrada.")

