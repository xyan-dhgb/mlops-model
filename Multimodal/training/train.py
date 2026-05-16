"""
train.py — Bước 5: Huấn luyện 2 giai đoạn Multimodal

FIX OOMKilled: Thay load_npy() (tải cả file vào RAM) bằng:
  1. download_file() → lưu .npy xuống disk (/tmp/train_data/)
  2. np.load(mmap_mode='r') → đọc qua memory-map, không chiếm RAM
  3. tf.data.Dataset generator → chỉ load từng batch vào RAM khi train

Đọc từ S3:
  s3://kltn-isic-2024-colab/splits/train/X_tab_train.npy
  s3://kltn-isic-2024-colab/splits/train/X_img_train.npy  (3.7 GiB)
  s3://kltn-isic-2024-colab/splits/train/y_train.npy
  s3://kltn-isic-2024-colab/splits/val/  (tương tự)
  s3://kltn-isic-2024-colab/preprocessed/encoders.pkl

Ghi lên S3:
  s3://kltn-isic-2024-colab/preprocessed/best_model_phase1.h5
  s3://kltn-isic-2024-colab/preprocessed/best_model_isic2024.h5
  s3://kltn-isic-2024-colab/preprocessed/training_history.pkl
  s3://kltn-isic-2024-colab/preprocessed/model_architecture.json
"""
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate,
    GlobalAveragePooling2D, BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

from augment import augment_image
from s3_utils import (
    download_file, load_pkl, save_pkl,
    upload_bytes, upload_file,
    S3_OUTPUT_BUCKET,
)

# ── Hyperparameters ──────────────────────────────────────────────────────
PHASE1_EPOCHS    = int(os.environ.get("PHASE1_EPOCHS",        "20"))
PHASE2_EPOCHS    = int(os.environ.get("PHASE2_EPOCHS",        "10"))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE",           "32"))
PHASE2_BATCH     = int(os.environ.get("PHASE2_BATCH_SIZE",    "16"))
OVERSAMPLE_RATIO = float(os.environ.get("OVERSAMPLE_RATIO",  "0.25"))
CLASS_W_MAL      = float(os.environ.get("CLASS_WEIGHT_MAL",  "1.2"))
FINE_TUNE_FROM   = int(os.environ.get("FINE_TUNE_FROM_LAYER", "300"))
PHASE1_LR        = float(os.environ.get("PHASE1_LR",         "1e-3"))
PHASE2_LR        = float(os.environ.get("PHASE2_LR",         "1e-4"))
IMAGE_SIZE       = int(os.environ.get("IMAGE_SIZE",           "224"))
IMAGE_SHAPE      = (IMAGE_SIZE, IMAGE_SIZE, 3)
TMP_DIR          = "/tmp/train_data"


# ── GPU setup ────────────────────────────────────────────────────────────
def setup_gpu():
    """Bật memory growth để TF không chiếm hết VRAM ngay khi khởi động."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU devices: {[g.name for g in gpus]}")
    return gpus


# ── Loss ─────────────────────────────────────────────────────────────────
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    fn.__name__ = "focal_loss"
    return fn


# ── Model ────────────────────────────────────────────────────────────────
def build_multimodal_model(tabular_dim: int, freeze_backbone: bool = True):
    image_input = Input(shape=IMAGE_SHAPE, name="image_input")
    backbone = EfficientNetB3(
        include_top=False, weights="imagenet",
        input_tensor=image_input, pooling=None,
    )
    backbone.trainable = not freeze_backbone
    if not freeze_backbone:
        for layer in backbone.layers[:FINE_TUNE_FROM]:
            layer.trainable = False

    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    tab_input = Input(shape=(tabular_dim,), name="tabular_input")
    y_tab = Dense(128, activation="relu")(tab_input)
    y_tab = BatchNormalization()(y_tab)
    y_tab = Dropout(0.3)(y_tab)
    y_tab = Dense(64, activation="relu")(y_tab)
    y_tab = Dropout(0.2)(y_tab)
    y_tab = Dense(32, activation="relu")(y_tab)

    combined = Concatenate()([x, y_tab])
    z = Dense(256, activation="relu")(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = Dense(128, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.2)(z)
    output = Dense(1, activation="sigmoid", name="output")(z)

    model = Model(inputs=[image_input, tab_input], outputs=output)
    return model, backbone


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=focal_loss(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


# ── tf.data helpers ──────────────────────────────────────────────────────
def build_train_dataset(
    X_img: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    tabular_dim: int,
    batch_size: int,
    oversample_ratio: float = 0.25,
) -> tf.data.Dataset:
    """
    tf.data.Dataset với oversampling + augmentation online.
    X_img là np.memmap — chỉ đọc từng ảnh khi cần, không copy toàn bộ vào RAM.
    """
    mal_idx = np.where(y == 1)[0]
    n_ben   = int((y == 0).sum())

    n_mal_target = int(n_ben * oversample_ratio / (1.0 - oversample_ratio))
    n_extra      = max(0, n_mal_target - len(mal_idx))
    extra_idx    = mal_idx[np.random.choice(len(mal_idx), size=n_extra, replace=True)]

    # Chỉ lưu danh sách idx — không copy ảnh
    all_idx = np.concatenate([np.arange(len(y)), extra_idx])
    np.random.shuffle(all_idx)

    n_total   = len(all_idx)
    n_mal_tot = int((y == 1).sum()) + n_extra
    print(f"  Train dataset: {len(y):,} gốc + {n_extra:,} oversample = {n_total:,} mẫu "
          f"(malignant {n_mal_tot/n_total:.1%})")

    # Giữ reference đến arrays (không copy)
    _X_img, _X_tab, _y = X_img, X_tab, y

    def generator():
        for idx in all_idx:
            img = _X_img[idx].copy()          # copy 1 ảnh (~600 KB)
            is_mal = bool(_y[idx] == 1)
            img = augment_image(img, strong=is_mal)
            yield (
                {
                    "image_input":   img.astype(np.float32),
                    "tabular_input": _X_tab[idx].astype(np.float32),
                },
                np.float32(_y[idx]),
            )

    return (
        tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    "image_input":   tf.TensorSpec(shape=IMAGE_SHAPE,     dtype=tf.float32),
                    "tabular_input": tf.TensorSpec(shape=(tabular_dim,),  dtype=tf.float32),
                },
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_val_dataset(
    X_img: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    tabular_dim: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Val dataset — không augment, không oversample, dùng memmap."""
    _X_img, _X_tab, _y = X_img, X_tab, y

    def generator():
        for i in range(len(_y)):
            yield (
                {
                    "image_input":   _X_img[i].astype(np.float32),
                    "tabular_input": _X_tab[i].astype(np.float32),
                },
                np.float32(_y[i]),
            )

    return (
        tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    "image_input":   tf.TensorSpec(shape=IMAGE_SHAPE,     dtype=tf.float32),
                    "tabular_input": tf.TensorSpec(shape=(tabular_dim,),  dtype=tf.float32),
                },
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# ── Download splits về disk ───────────────────────────────────────────────
def download_splits() -> dict:
    """Stream các file .npy từ S3 xuống disk — không đưa vào RAM."""
    os.makedirs(TMP_DIR, exist_ok=True)
    mapping = {
        "X_img_train": ("splits/train/X_img_train.npy", f"{TMP_DIR}/X_img_train.npy"),
        "X_tab_train": ("splits/train/X_tab_train.npy", f"{TMP_DIR}/X_tab_train.npy"),
        "y_train":     ("splits/train/y_train.npy",     f"{TMP_DIR}/y_train.npy"),
        "X_img_val":   ("splits/val/X_img_val.npy",     f"{TMP_DIR}/X_img_val.npy"),
        "X_tab_val":   ("splits/val/X_tab_val.npy",     f"{TMP_DIR}/X_tab_val.npy"),
        "y_val":       ("splits/val/y_val.npy",         f"{TMP_DIR}/y_val.npy"),
    }
    paths = {}
    for key, (s3_key, local_path) in mapping.items():
        print(f"  ↓ {s3_key} → {local_path}")
        download_file(s3_key, local_path, bucket=S3_OUTPUT_BUCKET)
        paths[key] = local_path
    return paths


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BƯỚC 5: Two-Phase Training  [memory-safe build]")
    print(f"  Bucket: s3://{S3_OUTPUT_BUCKET}/preprocessed/")
    print("=" * 60)

    gpus = setup_gpu()

    # 1. Download về disk (không load vào RAM)
    print("\n[1/5] Download splits về disk...")
    paths = download_splits()

    # 2. Mở qua memmap — X_img chỉ đọc từ disk khi cần
    print("\n[2/5] Load splits (memmap cho image arrays)...")
    X_img_train = np.load(paths["X_img_train"], mmap_mode="r")
    X_tab_train = np.load(paths["X_tab_train"])          # ~1 MB, OK in RAM
    y_train     = np.load(paths["y_train"])              # tiny
    X_img_val   = np.load(paths["X_img_val"],   mmap_mode="r")
    X_tab_val   = np.load(paths["X_tab_val"])
    y_val       = np.load(paths["y_val"])

    encoders    = load_pkl("preprocessed/encoders.pkl", bucket=S3_OUTPUT_BUCKET)
    tabular_dim = len(encoders["feature_cols"])
    print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | tabular_dim={tabular_dim}")
    print(f"  X_img_train: {X_img_train.shape}  dtype={X_img_train.dtype}  (memmap)")

    # 3. Class weight
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weight = {0: 1.0, 1: (n_neg / n_pos) * CLASS_W_MAL}
    print(f"\n  Class weights: {class_weight}")

    # 4. Build model
    print("\n[3/5] Build model...")
    model, backbone = build_multimodal_model(tabular_dim, freeze_backbone=True)
    model = compile_model(model, PHASE1_LR)
    upload_bytes(
        model.to_json().encode(),
        "preprocessed/model_architecture.json",
        bucket=S3_OUTPUT_BUCKET,
    )

    # 5. Build tf.data datasets (oversampling online, không copy mảng)
    print("\n[4/5] Xây tf.data datasets...")
    train_ds = build_train_dataset(
        X_img_train, X_tab_train, y_train, tabular_dim,
        batch_size=BATCH_SIZE, oversample_ratio=OVERSAMPLE_RATIO,
    )
    val_ds = build_val_dataset(
        X_img_val, X_tab_val, y_val, tabular_dim, batch_size=BATCH_SIZE,
    )

    # ── PHASE 1: Frozen backbone ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen backbone")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False, dir=TMP_DIR) as tmp1:
        phase1_local = tmp1.name

    cb1 = [
        EarlyStopping(monitor="val_auc", patience=5,
                      restore_best_weights=True, mode="max", verbose=1),
        ModelCheckpoint(phase1_local, monitor="val_auc",
                        save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=cb1,
        class_weight=class_weight,
        verbose=1,
    )

    upload_file(phase1_local, "preprocessed/best_model_phase1.h5",
                bucket=S3_OUTPUT_BUCKET)
    os.unlink(phase1_local)

    # ── PHASE 2: Unfreeze backbone ───────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-tune EfficientNetB3 từ layer {FINE_TUNE_FROM}")
    print("=" * 60)

    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_FROM]:
        layer.trainable = False
    print(f"  Layers mở đóng băng: "
          f"{sum(1 for l in backbone.layers if l.trainable)}/{len(backbone.layers)}")

    model = compile_model(model, PHASE2_LR)

    # Phase 2 dùng batch size nhỏ hơn + rebuild dataset
    train_ds_p2 = build_train_dataset(
        X_img_train, X_tab_train, y_train, tabular_dim,
        batch_size=PHASE2_BATCH, oversample_ratio=OVERSAMPLE_RATIO,
    )
    val_ds_p2 = build_val_dataset(
        X_img_val, X_tab_val, y_val, tabular_dim, batch_size=PHASE2_BATCH,
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False, dir=TMP_DIR) as tmp2:
        best_local = tmp2.name

    cb2 = [
        EarlyStopping(monitor="val_auc", patience=7,
                      restore_best_weights=True, mode="max", verbose=1),
        ModelCheckpoint(best_local, monitor="val_auc",
                        save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    h2 = model.fit(
        train_ds_p2,
        validation_data=val_ds_p2,
        epochs=PHASE2_EPOCHS,
        callbacks=cb2,
        class_weight=class_weight,
        verbose=1,
    )

    upload_file(best_local, "preprocessed/best_model_isic2024.h5",
                bucket=S3_OUTPUT_BUCKET)
    os.unlink(best_local)

    # Lưu history
    save_pkl(
        {"phase1": h1.history, "phase2": h2.history},
        "preprocessed/training_history.pkl",
        bucket=S3_OUTPUT_BUCKET,
    )

    print(f"\n[5/5] Model → s3://{S3_OUTPUT_BUCKET}/preprocessed/best_model_isic2024.h5")
    print("\nBước 5 hoàn thành!")


if __name__ == "__main__":
    main()
