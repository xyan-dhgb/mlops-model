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
def save_keras_model(model, s3_key: str, bucket: str = S3_OUTPUT_BUCKET):
    """Lưu model Keras (.h5) lên S3 qua file tạm."""
    import tensorflow as tf  # lazy — chỉ container có TF mới gọi hàm này
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        model.save(tmp.name)
        upload_file(tmp.name, s3_key, bucket)
    os.unlink(tmp.name)


def _patch_h5_config_for_tfkeras(h5_path: str) -> str:
    """Patch model_config JSON trong .h5 để tf_keras (Keras 2) load được model
    đã save bằng Keras 3.

    Keras 3 thêm các trường mà tf_keras không nhận dạng:
      - InputLayer.config['batch_shape']  → đổi thành 'batch_input_shape'
      - InputLayer.config['optional']     → bỏ đi
      - Dense/layer.config['quantization_config'] → bỏ đi
      - BatchNormalization.config renorm* → bỏ đi

    Trả về path file .h5 đã patch (file tạm mới).
    Nếu không cần patch, trả lại h5_path gốc.
    """
    import h5py
    import json
    import shutil

    with h5py.File(h5_path, "r") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            return h5_path  # không có config → bỏ qua
        config_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    config = json.loads(config_str)
    changed = [False]

    _BN_DROP  = {"renorm", "renorm_clipping", "renorm_momentum"}
    _ALL_DROP = {"quantization_config", "optional"}

    def _fix(obj):
        if isinstance(obj, dict):
            cls = obj.get("class_name", "")
            cfg = obj.get("config", {})

            if cls == "InputLayer":
                # batch_shape (Keras 3) → batch_input_shape (tf_keras)
                if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                    cfg["batch_input_shape"] = cfg.pop("batch_shape")
                    changed[0] = True
                # optional — không tồn tại trong tf_keras
                if cfg.pop("optional", None) is not None:
                    changed[0] = True

            if cls == "BatchNormalization":
                for k in list(cfg.keys()):
                    if k in _BN_DROP:
                        del cfg[k]
                        changed[0] = True

            # quantization_config xuất hiện ở mọi layer (Dense, v.v.)
            if "quantization_config" in cfg:
                del cfg["quantization_config"]
                changed[0] = True

            for v in obj.values():
                _fix(v)
        elif isinstance(obj, list):
            for item in obj:
                _fix(item)

    _fix(config)

    if not changed[0]:
        print("  [load_keras_model] model_config không cần patch")
        return h5_path

    # Tạo file .h5 mới với config đã patch
    patched_path = h5_path + ".patched.h5"
    shutil.copy2(h5_path, patched_path)
    with h5py.File(patched_path, "a") as f:
        f.attrs["model_config"] = json.dumps(config).encode("utf-8")

    print("  [load_keras_model] Đã patch model_config: "
          "batch_shape→batch_input_shape, bỏ optional/quantization_config/renorm*")
    return patched_path


def load_keras_model(s3_key: str, bucket: str = S3_OUTPUT_BUCKET,
                     custom_objects=None):
    """Load model Keras (.h5) từ S3.

    Chiến lược load triệt để Keras 2 ↔ Keras 3:
      1. Dùng tf_keras (Keras 2 standalone) nếu đã cài
      2. Trước khi load, patch model_config JSON trong .h5 bằng h5py
         để xử lý các trường Keras 3 không tương thích:
           - InputLayer: batch_shape → batch_input_shape, bỏ optional
           - Dense:      bỏ quantization_config
           - BatchNorm:  bỏ renorm*, renorm_clipping, renorm_momentum
    """
    # Chọn loader: ưu tiên tf_keras (Keras 2)
    load_model_fn = None
    try:
        import tf_keras
        load_model_fn = tf_keras.models.load_model
        print("  [load_keras_model] Dùng tf_keras (Keras 2 standalone)")
    except ImportError:
        import tensorflow as tf
        load_model_fn = tf.keras.models.load_model
        keras_ver = getattr(tf.keras, "__version__", "?")
        print(f"  [load_keras_model] Dùng tf.keras v{keras_ver}")

    # Download về disk
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        orig_path = tmp.name
    download_file(s3_key, orig_path, bucket)

    # Patch config JSON trong .h5 cho tương thích tf_keras
    patched_path = _patch_h5_config_for_tfkeras(orig_path)

    try:
        model = load_model_fn(patched_path, custom_objects=custom_objects)
    finally:
        _safe_unlink(orig_path)
        if patched_path != orig_path:
            _safe_unlink(patched_path)

    return model


def _safe_unlink(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass

