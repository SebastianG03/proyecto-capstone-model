from pathlib import Path

from app.core.config import BALL_MODEL_NAME, DEPTH_MODEL_NAME, GOAL_MODEL_NAME, PLAYER_MODEL_NAME

BASE_DIR = Path("app")
BASE_RES_DIR = BASE_DIR / "res"

OUTPUT_VIDEOS_DIR = BASE_RES_DIR / "output_videos"
OUTPUT_IMAGES_DIR = BASE_RES_DIR / "output_images"
OUTPUT_REPORTS_DIR = BASE_RES_DIR / "output_reports"
INPUT_VIDEOS_DIR = BASE_RES_DIR / "input_videos"
DATABASE_DIR = BASE_RES_DIR / "database"

MODELS_DIR = BASE_RES_DIR / "models"
BALL_MODEL_PATH = MODELS_DIR / BALL_MODEL_NAME
PLAYER_MODEL_PATH = MODELS_DIR / PLAYER_MODEL_NAME
MODEL_GOALS_PATH = MODELS_DIR / GOAL_MODEL_NAME
TROCR_PATH = MODELS_DIR / "trocr"
DEPTH_MODEL_PATH = MODELS_DIR / DEPTH_MODEL_NAME

def ensure_directories():
    """
    Asegura que las carpetas necesarias existan.
    """
    for directory in [
        OUTPUT_VIDEOS_DIR,
        OUTPUT_IMAGES_DIR,
        MODELS_DIR,
        TROCR_PATH,
        INPUT_VIDEOS_DIR,
        OUTPUT_REPORTS_DIR,
        DATABASE_DIR,
    ]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
