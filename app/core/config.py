import os
from pathlib import Path

ENV = os.environ.get('ENV', 'development')
DEBUG = os.environ.get('DEBUG', 'true').lower() in ('1','true','yes')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'debug' if DEBUG else 'error')
MODEL_PROFILE = os.environ.get('MODEL_PROFILE', 'light' if DEBUG else 'optimized')
PLAYER_IMG_DIR = os.environ.get('PLAYER_IMG_DIR', str(Path('app/res/player_images')))
# other default paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get('MODELS_DIR', BASE_DIR / 'app' / 'res' / 'models'))

def is_production():
    return ENV == 'production'
