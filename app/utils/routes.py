from pathlib import Path

from app.core.config import BALL_MODEL_NAME, DEPTH_MODEL_NAME, GOAL_MODEL_NAME, PLAYER_MODEL_NAME


BASE_DIR = Path("app")
BASE_RES_DIR = BASE_DIR / "res"

OUTPUT_VIDEOS_DIR = BASE_RES_DIR / "output_videos"
OUTPUT_IMAGES_DIR = BASE_RES_DIR / "output_images"
OUTPUT_REPORTS_DIR = BASE_RES_DIR / "output_reports"
INPUT_VIDEOS_DIR = BASE_RES_DIR / "input_videos"
DATABASE_DIR = BASE_RES_DIR / "database"
DATASETS_DIR = BASE_RES_DIR / "datasets"

ANOTATED_VIDEOS_DIR = OUTPUT_VIDEOS_DIR / "anotated"
ANOTATED_OUTPUT_IMAGES = OUTPUT_IMAGES_DIR / "anotated_images"
METRICS_DIR = OUTPUT_REPORTS_DIR / "metrics"
DETECTED_OBJECTS_METRICS_DIR = OUTPUT_REPORTS_DIR / "detected_objects_metrics"
MEMORY_TRACKER_DIR = OUTPUT_REPORTS_DIR / "memory_tracker"
MODELS_DIR = BASE_RES_DIR / "models"
MODELS_BACKUP_DIR = BASE_RES_DIR / "models_backup"

BALL_MODEL_PATH = MODELS_DIR / BALL_MODEL_NAME
PLAYER_MODEL_PATH = MODELS_DIR / PLAYER_MODEL_NAME
MODEL_GOALS_PATH = MODELS_DIR / GOAL_MODEL_NAME
TROCR_PATH = MODELS_DIR / "trocr"
DEPTH_MODEL_PATH = MODELS_DIR / DEPTH_MODEL_NAME

TRACKER_CONFIG_PATH = MODELS_DIR / "tracker"
BYTETRACK_CONFIG_PATH = TRACKER_CONFIG_PATH / "bytetrack.yaml"
PLAYER_XGB_MODEL = MODELS_DIR / "player_XGB"
PLAYER_YOLO_DATA = MODELS_DIR / "YOLO_pickles"
RETRAINED_MODELS = MODELS_DIR / "retrained_models"

PLAYER_CUSTOM_DATASET = DATASETS_DIR / "custom_player_dataset"
BALL_CUSTOM_DATASET = DATASETS_DIR / "custom_ball_dataset"
CUSTOM_MODELS = DATASETS_DIR / "custom_models"

def ensure_directories():
    """
    Asegura que las carpetas necesarias existan.
    """
    for directory in [
        OUTPUT_VIDEOS_DIR,
        OUTPUT_IMAGES_DIR,
        MODELS_DIR,
        DATASETS_DIR,
        TROCR_PATH,
        INPUT_VIDEOS_DIR,
        OUTPUT_REPORTS_DIR,
        DATABASE_DIR,
        ANOTATED_VIDEOS_DIR,
        ANOTATED_OUTPUT_IMAGES,
        METRICS_DIR,
        MEMORY_TRACKER_DIR,
        DETECTED_OBJECTS_METRICS_DIR,
        TRACKER_CONFIG_PATH,
        PLAYER_XGB_MODEL,
        PLAYER_YOLO_DATA,
        RETRAINED_MODELS,
        PLAYER_CUSTOM_DATASET,
        BALL_CUSTOM_DATASET,
        MODELS_BACKUP_DIR,
        CUSTOM_MODELS
    ]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
