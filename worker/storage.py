import os
import uuid
import boto3
from botocore.config import Config
import requests

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_DOMAIN = os.getenv("R2_PUBLIC_DOMAIN", "")


def get_r2_client():
    if not R2_ACCOUNT_ID:
        raise ValueError("R2_ACCOUNT_ID is not set")
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def download_audio(url: str, local_path: str) -> str:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return local_path


def upload_to_r2(local_path: str, content_type: str = "audio/wav", key: str = None) -> str:
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        raise ValueError("R2 credentials or bucket name not configured")

    if key is None:
        ext = os.path.splitext(local_path)[1] or ".wav"
        key = f"tts-output/{uuid.uuid4()}{ext}"

    client = get_r2_client()
    extra_args = {"ContentType": content_type}
    client.upload_file(local_path, R2_BUCKET_NAME, key, ExtraArgs=extra_args)

    if R2_PUBLIC_DOMAIN:
        return f"{R2_PUBLIC_DOMAIN.rstrip('/')}/{key}"

    return f"https://{R2_BUCKET_NAME}.{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{key}"
