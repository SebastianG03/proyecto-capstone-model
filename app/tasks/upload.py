import datetime
from io import BytesIO
import boto3
from decouple import config
from botocore.client import Config
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed

s3_client = boto3.client(
    "s3",
    endpoint_url=config("VIDEOS_S3_ENDPOINT"),
    aws_access_key_id=config("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=config("R2_SECRET_ACCESS_KEY"),
    region_name="auto",
    config=Config(signature_version="s3v4")
)

def upload_file(
    match_id: int,
    player_id: str,
    filename: str,
    file_bytes: bytes,
):
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    key = f"{match_id}/{timestamp}_{filename}"
    upload(key, file_bytes)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10))
def upload(key: str, file_bytes: bytes, file_type: str = "image/png"):
    try:
        s3_client.upload_fileobj(
            Fileobj=BytesIO(file_bytes),
            Bucket=config("BUCKET"),
            Key=key,
            ExtraArgs={"ContentType": file_type}
        )

        print("Archivo subido correctamente:", key)

    except Exception as e:
        print("Error uploading to R2:", e)
        raise e
