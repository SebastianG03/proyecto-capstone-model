from pathlib import Path

BASE_DIR = Path("app")
BASE_RES_DIR = BASE_DIR / "res"
OUTPUT_VIDEOS_DIR = BASE_RES_DIR / "output_videos"
OUTPUT_IMAGES_DIR = BASE_RES_DIR / "output_images"
OUTPUT_REPORTS_DIR = BASE_RES_DIR / "output_reports"
MODELS_DIR = BASE_RES_DIR / "models"
INPUT_VIDEOS_DIR = BASE_RES_DIR / "input_videos"

def ensure_directories():
    """
    Asegura que las carpetas necesarias existan.
    """
    for directory in [
        OUTPUT_VIDEOS_DIR,
        OUTPUT_IMAGES_DIR,
        MODELS_DIR,
        INPUT_VIDEOS_DIR,
        OUTPUT_REPORTS_DIR
    ]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)