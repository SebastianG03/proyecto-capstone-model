from pathlib import Path
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, wait_incrementing
from app.logger import info_logger, error_logger

from app.core.config import (
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    VIDEO_BUCKET,
    VIDEOS_S3_ENDPOINT,
)
from app.entities.utils.singleton import Singleton


class R2Downloader(metaclass=Singleton):
    def __init__(
        self,
    ):
        config = Config(
            read_timeout=800,
            connect_timeout=50,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self.s3 = boto3.client(
            "s3",
            endpoint_url=VIDEOS_S3_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=config,
        )
        self.bucket = VIDEO_BUCKET

    def build_destination_path(self, key: str, base_dir: str = "./tmp") -> Path:
        """
        Construye un Path válido para guardar el archivo usando pathlib.
        Extrae automáticamente el nombre del archivo desde el key.
        """
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)

        filename = Path(key).name
        info_logger.info(f"Construyendo ruta de destino para {filename} en {base_dir}...")
        return base / filename

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_incrementing(start=0, increment=2, max=10),
        reraise=True,
        )
    def stream_download(
        self, key: str, destination_path: str, chunk_size=1024 * 1024 * 16
    ):
        """
        Descarga el archivo en chunks (16 MB por defecto).
        Soporta archivos grandes (+5GB).
        """
        try:
            info_logger.info(f"Descargando {key} a {destination_path}...")
            with open(destination_path, "wb") as f:
                info_logger.info(f"Abriendo conexión a R2 para el objeto {key}...")
                obj = self.s3.get_object(Bucket=self.bucket, Key=key)
                info_logger.info(f"Iniciando descarga en chunks de {chunk_size} bytes...")
                body = obj["Body"]
                info_logger.info("Descargando...")
                while True:
                    chunk = body.read(chunk_size)
                    info_logger.info(f"Descargados {f.tell()} bytes...")
                    if not chunk:
                        info_logger.info("Descarga completada.")
                        break
                    f.write(chunk)
                    f.flush()
                    info_logger.info(f"Bytes recibidos: {f.tell()}")
        except (BotoCoreError, ClientError, ReadTimeoutError) as e:
            error_logger.error(f"Error descargando {key} desde R2: {e}")
            Path(destination_path).unlink(missing_ok=True)
            raise e
        except Exception as e:
            error_logger.error(f"Error descargando {key} desde R2: {e}")
            Path(destination_path).unlink(missing_ok=True)
            raise e
