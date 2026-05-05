from pathlib import Path
from requests import get
from app.core.config import DOWNLOAD_MODEL_URL

model_url = DOWNLOAD_MODEL_URL


def model_exists(model_path: Path) -> bool:
    return model_path.exists() and model_path.is_file()


def prepare_model(model_path: Path, source_path: Path) -> None:
    if model_exists(model_path):
        return
    if not source_path.exists() or not source_path.is_file():
        source_path.mkdir(parents=True, exist_ok=True)

    r = get(str(model_url), timeout=300)

    with open(model_path, "wb") as f:
        f.write(r.content)
