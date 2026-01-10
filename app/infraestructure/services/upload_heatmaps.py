from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from app.entities.models import PlayerState
from app.infraestructure.services.upload_service import upload_file
from app.utils.routes import OUTPUT_VIDEOS_DIR
from sqlalchemy.orm import Session
from app.logger import debug_logger

def upload_heatmaps_for_extracted_players(
    db: Session, match_id: int, extracted_player_ids: set
) -> dict[int, str]:

    """
    Sube los heatmaps de los jugadores extraídos a AWS S3.

    Parameters:
    db (Session): Sesión de la base de datos.
    match_id (int): ID del partido.
    extracted_player_ids (set): Conjunto de IDs de los jugadores extraídos.

    Returns:
    dict[int, str]: Diccionario con los IDs de los jugadores como claves y las
    claves como valores correspondientes a los nombres de los archivos subidos en
    AWS S3.
    """
    try:
        base_path = OUTPUT_VIDEOS_DIR
        players_path = base_path / "players"
        files_in_folder = list(players_path.glob("heatmap_player_*.png"))
        print(f"Archivos encontrados en players: {[f.name for f in files_in_folder]}")

        if not files_in_folder:
            print("No se encontraron archivos de heatmaps en la carpeta de players.")
            return {}

        jobs = []
        for file in files_in_folder:
            home_file = players_path / file.name
            if not home_file.exists() or home_file.stat().st_size == 0:
                print(f"Archivo inválido o vacío: {file.name}")
                continue

            file_bytes = home_file.read_bytes()
            player_id = file.stem.split("_")[2]

            if int(player_id) not in extracted_player_ids:
                continue

            jobs.append({
                "player_id": player_id,
                "filename": home_file.name,
                "file_bytes": file_bytes
            })

        if not jobs:
            print("Nada que subir.")
            return {}

        uploaded_keys: dict = {}
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = [
                exe.submit(upload_file, match_id, j["player_id"], j["filename"], j["file_bytes"])
                for j in jobs
            ]
            for fut in as_completed(futures):
                try:
                    key = fut.result()
                    if key:
                        debug_logger.debug(f"[UPLOAD HEATMAP] Heatmap subido: {key}")
                        uploaded_keys[key["player_id"]] = key["key"]
                except Exception as exc:
                    print(f"Error subiendo heatmap: {exc}")

        print("Subida de heatmaps completada.")
        return uploaded_keys

    except Exception as e:
        print(f"Error al subir heatmaps: {e}")
        raise e
