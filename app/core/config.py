import os
from decouple import config


# S3
S3_API = str(config("S3_API"))
VIDEOS_S3_ENDPOINT = str(config("VIDEOS_S3_ENDPOINT"))
BUCKET = str(config("BUCKET"))
VIDEO_BUCKET = str(config("VIDEO_BUCKET"))

# API
STATS_NOTIFY_URL = str(config("STATS_NOTIFY_URL"))

# R2
R2_ACCESS_TOKEN = str(config("R2_ACCESS_TOKEN"))
R2_ACCESS_KEY_ID = str(config("R2_ACCESS_KEY_ID"))
R2_SECRET_ACCESS_KEY = str(config("R2_SECRET_ACCESS_KEY"))
R2_ACCOUNT_ID = str(config("R2_ACCOUNT_ID"))
PUBLIC_URL = str(config("PUBLIC_URL"))
# SETTINGS
DEBUG = config("DEBUG", default=False, cast=bool)
MAX_EMPTY_BATCHES = config("MAX_EMPTY_BATCHES", default=5, cast=int)
MAX_PROCESSING_TIME = config("MAX_PROCESSING_TIME", default=-1, cast=int)
DOWNLOAD_MODEL_URL = str(config("DOWNLOAD_MODEL_URL"))
USE_PARALLEL_IO = config("USE_PARALLEL_IO", default=True, cast=bool)
MODEL_USE_HALF_PRECISION = config("MODEL_USE_HALF_PRECISION", default=False, cast=bool)
BATCH_SIZE = config("BATCH_SIZE", default=30, cast=int)

# MODELS    
BALL_MODEL_NAME = str(config("BALL_MODEL_NAME"))
PLAYER_MODEL_NAME = str(config("PLAYER_MODEL_NAME"))
DEPTH_MODEL_NAME = str(config("DEPTH_MODEL_NAME", cast=str))
GOAL_MODEL_NAME =  str(config("GOAL_MODEL_NAME", cast=str))

if not GOAL_MODEL_NAME or not BALL_MODEL_NAME or not PLAYER_MODEL_NAME or not DEPTH_MODEL_NAME:
    model_lacked = [
        "GOAL_MODEL_NAME" if not GOAL_MODEL_NAME else "",
        "BALL_MODEL_NAME" if not BALL_MODEL_NAME else "",
        "PLAYER_MODEL_NAME" if not PLAYER_MODEL_NAME else "",
        "DEPTH_MODEL_NAME" if not DEPTH_MODEL_NAME else "",
    ]
    raise Exception("Missing model name in .env: " + ", ".join(model_lacked))

os.environ["USE_CUDA"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
