import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed   # opcional
from app.entities.models import PlayerState
from app.tasks.upload import upload_file
from app.utils.routes import OUTPUT_VIDEOS_DIR
from sqlalchemy.orm import Session

def upload_heatmaps_for_extracted_players(db: Session, match_id: int, extracted_player_ids: set):
    """
    Sube todos los heatmaps generados para los jugadores que se extrajeron.
    Filtra por player_id y sube solo los heatmaps que coincidan con los jugadores.
    """
    try:
        base_path   = OUTPUT_VIDEOS_DIR
        players_path = base_path / "players"
        files_in_folder = list(players_path.glob("heatmap_player_*.png"))
        print(f"Archivos encontrados en players: {[f.name for f in files_in_folder]}")

        if not files_in_folder:
            print("No se encontraron archivos de heatmaps en la carpeta de players.")
            return

        jobs = []
        id_map = {}

        for file in files_in_folder:
            home_file = players_path / file.name
            if not home_file.exists() or home_file.stat().st_size == 0:
                print(f"Archivo inválido o vacío: {file.name}")
                continue

            # leer bytes una sola vez
            file_bytes = home_file.read_bytes()

            # extraer id del nombre
            id_str = file.stem.split("_")[2]
            player_id = id_str

            jobs.append({
                "player_id": player_id,
                "filename" : home_file.name,
                "file_bytes": file_bytes
            })
            id_map[player_id] = home_file.name

        if not jobs:
            print("Nada que subir.")
            return

        results = []
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = [
                exe.submit(upload_file,
                    match_id,
                    j["player_id"],
                    j["filename"],
                    j["file_bytes"]) for j in jobs]
            for fut in as_completed(futures):
                try:
                    key = fut.result()
                    results.append(key)
                except Exception as exc:
                    print(f"Error subiendo heatmap: {exc}")
                    results.append(None)

        print("Subida de heatmaps completada.")

    except Exception as e:
        print(f"Error al subir heatmaps: {e}")
        raise
