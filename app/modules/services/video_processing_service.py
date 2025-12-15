import pathlib
import time
from typing import Generator, List

import cv2
from cv2.typing import MatLike

from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import PlayerStateModel
from app.logger import debug_logger

def read_video(video_path: str, batch_size: int = 16) -> Generator[List[tuple[MatLike, float]]]:
    print(f"Abriendo video para lectura: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_time = time.time()
        frame_count = 0

        while frame_count < total_frames:
            batch = []
            now = time.time()
            dt = now - last_time
            last_time = now

            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                print("Frame válido obtenido.")
                frame_count += 1
                print(f"Leyendo frame {frame_count + 1}...")
                print(f"Frame leído: {'sí' if ret else 'no'}")
                batch.append((frame, dt))
                print(f"Tiempo desde último frame: {dt:.4f} segundos")
                
                if frame_count >= total_frames:
                    break
            
            print(f"Tamaño del batch actual: {len(batch)}")
            if batch:
                yield batch
            else:
                break
    except Exception as e:
        print(f"Error leyendo el video {video_path}: {e}")
        raise e
    finally:
        cap.release()

def extract_player_images(
    frame: MatLike,
    frame_index: int,
    player: PlayerStateModel,
    output_folder: str,
    player_image_counts: dict, 
    last_frame_taken: dict,
    images_per_player: int = 3,
    frame_skip: int = 5,
):
    """
    Extrae imágenes de torso/cara de jugadores PARA EL FRAME ACTUAL.
    Esta función debe llamarse en cada iteración del lector de video.

    :param frame: Frame actual del video
    :param frame_index: Índice del frame actual
    :param records: Lista de records del frame actual (track_id, bbox, etc.)
    :param output_folder: Carpeta de salida
    :param images_per_player: Máximo de imágenes por jugador
    :param frame_skip: Diferencia mínima entre capturas del mismo jugador
    :param player_image_counts: Mapa acumulado {player_id: cantidad}
    :param last_frame_taken: Mapa {player_id: ultimo_frame_guardado}
    """
    try:
        debug_logger.debug(f"[Extract Player Images] Extrayendo imagen de jugador {player.to_dict().get('player_id', None)} en frame {frame_index}...")
        folder = pathlib.Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)

        if player_image_counts is None:
            debug_logger.debug("[Extract Player Images] Inicializando player_image_counts...")
            player_image_counts = {}

        if last_frame_taken is None:
            debug_logger.debug("[Extract Player Images] Inicializando last_frame_taken...")
            last_frame_taken = {}

        debug_logger.debug(f"[Extract Player Images] Dimensiones del frame: {frame.shape}")
        h, w = frame.shape[:2]

        # Procesar solo records del frame actual
        debug_logger.debug(f"[Extract Player Images] Procesando jugador: {player.to_dict()}")
        record = player.to_dict()
        player_id: int = int(record.get("player_id", -1))
        player_team = record.get("team", "unknown")
        player_color = record.get("color", "unknown")
        bbox = player.get_bbox()
        
        if player_id == -1:
            debug_logger.debug("[Extract Player Images] Player ID inválido.")
            return player_image_counts, last_frame_taken, None

        if bbox is None or len(bbox) != 4:
            debug_logger.debug("[Extract Player Images] No hay bounding box para el jugador.")
            return player_image_counts, last_frame_taken, None

        # Máximo por jugador
        count = player_image_counts.get(player_id, 0)
        if count >= images_per_player:
            debug_logger.debug("[Extract Player Images] Máximo de imágenes alcanzado para el jugador.")
            return player_image_counts, last_frame_taken, None

        # Saltar frames cercanos
        last_f = last_frame_taken.get(player_id, -frame_skip - 1)
        if frame_index - last_f < frame_skip:
            debug_logger.debug("[Extract Player Images] Saltando frame por frame_skip.")
            return player_image_counts, last_frame_taken, None
        
        h, w = frame.shape[:2]

        # Validar bounding box
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            debug_logger.debug("[Extract Player Images] Bounding box inválido.")
            return player_image_counts, last_frame_taken, None

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            debug_logger.debug("[Extract Player Images] Bounding box muy pequeño.")
            return player_image_counts, last_frame_taken, None
        
        torso_y2 = y1 + int((y2 - y1) * 0.6)
        crop = frame[y1:torso_y2, x1:x2]
        
        if crop.size == 0:
            debug_logger.debug("[Extract Player Images] Crop resultó en una imagen vacía.")
            return player_image_counts, last_frame_taken, None

        # Guardar imagen
        filename = folder / f"player_{player_id}_team_{player_team}_color_{player_color}_img_{count+1}_frame_{frame_index}.png"
        debug_logger.debug(f"[Extract Player Images] Guardando imagen en {filename}...")
        cv2.imwrite(str(filename), crop)
        player_image_counts.update({player_id: count + 1})
        last_frame_taken.update({player_id: frame_index})

        debug_logger.debug("[Extract Player Images] Imagen guardada.")

        return player_image_counts, last_frame_taken, player_id
    except Exception as e:
        print(f"Error extrayendo imagen de jugador { player.to_dict().get('player_id', None) } en frame {frame_index}: {e}")
        raise e