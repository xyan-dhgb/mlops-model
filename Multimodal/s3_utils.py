"""
s3_utils.py — Tiện ích S3 dùng chung cho tất cả container
Được COPY vào mỗi container và import trực tiếp.

Buckets:
  Input : s3://kltn-isic-2024-challenge/isic-2024-challenge/
  Output: s3://kltn-isic-2024-colab/
    ├── raw/
    │   ├── metadata.csv
    │   └── images/<isic_id>.jpg     ← ảnh trích xuất từ HDF5 (byte gốc)
    ├── preprocessed/
    │   ├── metadata_clean.csv
    │   ├── encoders.pkl
    │   └── images_resized/<isic_id>.png  ← sau CLAHE+Gaussian+Contrast
    ├── features/
    │   ├── X_tabular.npy
    │   ├── X_images.npy
    │   └── y_labels.npy
    └── splits/
        ├── train/  X_tab_train.npy  X_img_train.npy  y_train.npy
        ├── val/    X_tab_val.npy    X_img_val.npy    y_val.npy
        └── test/   X_tab_test.npy   X_img_test.npy   y_test.npy
"""
import io
import os
import pickle
import tempfile
import numpy as np
import boto3
from botocore.exceptions import ClientError

# ── Khởi tạo client từ biến môi trường (IAM Role trên EKS hoặc key) ────
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN") or None,
    )

S3_INPUT_BUCKET  = os.environ.get("S3_INPUT_BUCKET",  "kltn-isic-2024-challenge")
S3_INPUT_PREFIX  = os.environ.get("S3_INPUT_PREFIX",  "isic-2024-challenge")
S3_OUTPUT_BUCKET = os.environ.get("S3_OUTPUT_BUCKET", "kltn-isic-2024-colab")


# ── Generic upload / download ────────────────────────────────────────────
def upload_bytes(data: bytes, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    s3 = get_s3_client()
    s3.put_object(Body=data, Bucket=bucket, Key=s3_key)
    print(f"  ↑ s3://{bucket}/{s3_key}  ({len(data):,} bytes)")


def download_bytes(s3_key: str, bucket: str = S3_OUTPUT_BUCKET) -> bytes:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    data = obj["Body"].read()
    print(f"  ↓ s3://{bucket}/{s3_key}  ({len(data):,} bytes)")
    return data


def upload_file(local_path: str, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    s3 = get_s3_client()
    s3.upload_file(local_path, bucket, s3_key)
    size = os.path.getsize(local_path)
    print(f"  ↑ s3://{bucket}/{s3_key}  ({size:,} bytes)")


def download_file(s3_key: str, local_path: str, bucket: str = S3_OUTPUT_BUCKET):
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    s3.download_file(bucket, s3_key, local_path)
    size = os.path.getsize(local_path)
    print(f"  ↓ s3://{bucket}/{s3_key} → {local_path}  ({size:,} bytes)")


def s3_key_exists(s3_key: str, bucket: str = S3_OUTPUT_BUCKET) -> bool:
    try:
        get_s3_client().head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError:
        return False


def list_s3_keys(prefix: str, bucket: str = S3_OUTPUT_BUCKET) -> list[str]:
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ── NumPy helpers ────────────────────────────────────────────────────────
def save_npy(array: np.ndarray, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, array)
        upload_file(tmp.name, s3_key, bucket)
    os.unlink(tmp.name)


def load_npy(s3_key: str, bucket: str = S3_OUTPUT_BUCKET) -> np.ndarray:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        download_file(s3_key, tmp.name, bucket)
        array = np.load(tmp.name, allow_pickle=False)
    os.unlink(tmp.name)
    return array


# ── Pickle helpers ───────────────────────────────────────────────────────
def save_pkl(obj, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    upload_bytes(pickle.dumps(obj), s3_key, bucket)


def load_pkl(s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    return pickle.loads(download_bytes(s3_key, bucket))


# ── CSV helpers ──────────────────────────────────────────────────────────
def save_csv(df, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    import pandas as pd
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    upload_bytes(buf.getvalue().encode(), s3_key, bucket)


def load_csv(s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    import pandas as pd
    data = download_bytes(s3_key, bucket)
    return pd.read_csv(io.BytesIO(data))


# ── Image helpers ────────────────────────────────────────────────────────
def save_png(img_array: np.ndarray, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    """Lưu numpy HWC uint8 lên S3 dạng PNG."""
    from PIL import Image
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    upload_bytes(buf.getvalue(), s3_key, bucket)


def load_png(s3_key: str, bucket: str = S3_OUTPUT_BUCKET) -> np.ndarray:
    from PIL import Image
    data = download_bytes(s3_key, bucket)
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


# ── Keras model helpers ──────────────────────────────────────────────────
import tensorflow as tf

class _Keras2BatchNorm(tf.keras.layers.BatchNormalization):
    """Compatibility shim: silently drops Keras-2-only kwargs
    (renorm, renorm_clipping, renorm_momentum) so .h5 models saved with
    old tf.keras can be loaded by Keras 3 without ValueError."""
    def __init__(self, **kwargs):
        for k in ("renorm", "renorm_clipping", "renorm_momentum"):
            kwargs.pop(k, None)
        super().__init__(**kwargs)


def save_keras_model(model, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    """Lưu model Keras (.h5) lên S3 qua file tạm."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        model.save(tmp.name)
        upload_file(tmp.name, s3_key, bucket)
    os.unlink(tmp.name)


def load_keras_model(s3_key: str, bucket: str = S3_OUTPUT_BUCKET,
                     custom_objects=None):
    # Inject _Keras2BatchNorm shim so .h5 models saved with Keras 2
    # (which stored renorm/renorm_clipping/renorm_momentum) load cleanly
    # under Keras 3 that no longer accepts those kwargs.
    merged = {"BatchNormalization": _Keras2BatchNorm}
    if custom_objects:
        merged.update(custom_objects)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        download_file(s3_key, tmp.name, bucket)
        model = tf.keras.models.load_model(tmp.name, custom_objects=merged)
    os.unlink(tmp.name)
    return model
