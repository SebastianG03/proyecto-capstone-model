from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import app.entities.utils.tools_context as context 
from app.infraestructure.services.upload_service import upload_file
from sqlalchemy.orm import Session
from app.logger import debug_logger, info_logger
from app.core.config import DEBUG


def upload_heatmaps_for_extracted_players(
    db: Session, match_id: int
) -> dict[int, str]:
    """
    Sube los heatmaps de los jugadores extraidos a AWS S3.

    Parameters:
    db (Session): Sesion de la base de datos.
    match_id (int): ID del partido.
    extracted_player_ids (set): Conjunto de IDs de los jugadores extraidos.

    Returns:
    dict[int, str]: Diccionario con los IDs de los jugadores como claves y las
    claves como valores correspondientes a los nombres de los archivos subidos en
    AWS S3.
    """
    heatmaps = context.analysis_context.tools.heatmap_points.get_all()
    files_in_folder: List[Path] = [Path(f'{map.path}') for map in heatmaps if Path(f'{map.path}').is_file()] 
    print(f"Archivos encontrados en players: {[f.name for f in files_in_folder]}")

    try:
        if not files_in_folder:
            info_logger.info("[Upload Heatmaps] No se encontraron archivos de heatmaps en la carpeta de players.")
            return {}

        jobs = []
        for file in files_in_folder:
            if not file.exists() or file.stat().st_size == 0:
                info_logger.info(f"[Upload Heatmaps] Archivo invalido o vacio: {file.name}")
                continue

            file_bytes = file.read_bytes()
            player_id = file.stem.split("_")[2]

            jobs.append({
                "player_id": player_id,
                "filename": file.name,
                "file_bytes": file_bytes,
            })

        if not jobs:
            info_logger.info("[Upload Heatmaps] Nada que subir.")
            return {}

        uploaded_keys: dict = {}
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = [
                exe.submit(
                    upload_file,
                    match_id,
                    j["player_id"],
                    j["filename"],
                    j["file_bytes"],
                )
                for j in jobs
            ]
            for fut in as_completed(futures):
                try:
                    key = fut.result()
                    if key:
                        debug_logger.debug(f"[UPLOAD HEATMAP] Heatmap subido: {key}")
                        heatmap = context.analysis_context.tools.heatmap_points.get_by_player_id(int(key["player_id"]))
                        context.analysis_context.tools.heatmap_points.patch(
                            int(f'{heatmap.id}'),
                            {"path": key["key"]})
                        debug_logger.debug(f"[UPLOAD HEATMAP] Heatmap actualizado: {key}")
                        uploaded_keys[key["player_id"]] = key["key"]
                except Exception as exc:
                    print(f"Error subiendo heatmap: {exc}")

        print("Subida de heatmaps completada.")
        return uploaded_keys

    except Exception as e:
        print(f"Error al subir heatmaps: {e}")
        raise e
    finally:
        if DEBUG:
            debug_logger.debug("[UPLOAD HEATMAP] En debug, no se limpian los archivos.")
        else:
            for file in files_in_folder:
                file.unlink()
