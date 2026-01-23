from pathlib import Path
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

from app.core.config import R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, VIDEO_BUCKET, VIDEOS_S3_ENDPOINT
from app.entities.utils.singleton import Singleton

class R2Downloader(metaclass=Singleton):
    def __init__(self,):
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
        print(f"Construyendo ruta de destino para {filename} en {base_dir}...")
        return base / filename

    def stream_download(self, key: str, destination_path: str, chunk_size=1024*1024*16):
        """
        Descarga el archivo en chunks (16 MB por defecto).
        Soporta archivos grandes (+5GB).
        """
        try:
            print(f"Descargando {key} a {destination_path}...")
            with open(destination_path, "wb") as f:
                print(f"Abriendo conexión a R2 para el objeto {key}...")
                obj = self.s3.get_object(Bucket=self.bucket, Key=key)
                print(f"Iniciando descarga en chunks de {chunk_size} bytes...")
                body = obj["Body"]
                print(f"Descargando...")
                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        print(f"Descarga completada.")
                        break
                    f.write(chunk)
                    f.flush()
        except (BotoCoreError, ClientError, ReadTimeoutError) as e:
            print(f"Error descargando {key} desde R2: {e}")
            Path(destination_path).unlink(missing_ok=True)
            raise e
        except Exception as e:
            print(f"Error descargando {key} desde R2: {e}")
            Path(destination_path).unlink(missing_ok=True)
            raise e
