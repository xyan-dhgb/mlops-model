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

MLflow:
  - Log params + metrics mỗi epoch (cả 2 phase) vào tracking server
  - Đăng ký model tốt nhất (phase 2) vào MLflow Model Registry
"""
import os
import sys
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
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback,
)

import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature

from augment import augment_image
from s3_utils import (
    download_file, load_pkl, save_pkl,
    upload_bytes, upload_file,
    s3_key_exists,
    S3_OUTPUT_BUCKET,
)

try:
    from ml_metrics import MetricsServer, record_epoch_metrics
except ImportError:
    pass

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

# ── MLflow config ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI",  "https://kltn-mlflow-ui.tech")
MLFLOW_EXPERIMENT    = os.environ.get("MLFLOW_EXPERIMENT",    "isic2024-multimodal")
MLFLOW_MODEL_NAME    = os.environ.get("MLFLOW_MODEL_NAME",    "isic2024-multimodal-best")


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


# ── MLflow per-epoch callback ────────────────────────────────────────────
class MlflowEpochLogger(Callback):
    """Log metrics lên MLflow sau mỗi epoch."""

    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase  # "phase1" hoặc "phase2"

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        step = epoch
        metrics = {}
        for k, v in logs.items():
            # prefix để phân biệt phase1/phase2 trong cùng 1 run
            metrics[f"{self.phase}/{k}"] = float(v)
        mlflow.log_metrics(metrics, step=step)


class PrometheusEpochLogger(Callback):
    """Log metrics lên Prometheus sau mỗi epoch."""

    def __init__(self, total_epochs, phase: str):
        super().__init__()
        self.total_epochs = total_epochs
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        try:
            record_epoch_metrics(
                epoch=epoch,
                total_epochs=self.total_epochs,
                train_loss=float(logs.get("loss", 0.0)),
                val_loss=float(logs.get("val_loss", 0.0)),
                val_auc=float(logs.get("val_auc", 0.0)),
                val_pauc=float(logs.get("val_pauc", 0.0)),
                model_name="isic-multimodal",
                modality=self.phase
            )
        except NameError:
            pass


# ── tf.data helpers ──────────────────────────────────────────────────────
def build_train_dataset(
    X_img: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    tabular_dim: int,
    batch_size: int,
    oversample_ratio: float = 0.25,
) -> tuple[tf.data.Dataset, int]:
    """
    tf.data.Dataset với oversampling + augmentation online.
    X_img là np.memmap — chỉ đọc từng ảnh khi cần, không copy toàn bộ vào RAM.

    Returns:
        (dataset, steps_per_epoch) — dataset lặp vô hạn (.repeat()),
        steps_per_epoch cần truyền vào model.fit() để Keras biết khi nào hết 1 epoch.
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
    steps_per_epoch = int(np.ceil(n_total / batch_size))
    print(f"  Train dataset: {len(y):,} gốc + {n_extra:,} oversample = {n_total:,} mẫu "
          f"(malignant {n_mal_tot/n_total:.1%})")
    print(f"  steps_per_epoch = {steps_per_epoch}")

    # Giữ reference đến arrays (không copy)
    _X_img, _X_tab, _y = X_img, X_tab, y
    _all_idx = all_idx  # snapshot để closure an toàn

    def generator():
        """Generator lặp vô tận — .repeat() ở ngoài hoặc shuffle lại mỗi epoch."""
        while True:
            # Shuffle lại mỗi lần lặp để mỗi epoch khác nhau
            epoch_idx = _all_idx.copy()
            np.random.shuffle(epoch_idx)
            for idx in epoch_idx:
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

    ds = (
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
    return ds, steps_per_epoch


def build_val_dataset(
    X_img: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    tabular_dim: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, int]:
    """Val dataset — không augment, không oversample, dùng memmap.

    Returns:
        (dataset, validation_steps) — dataset lặp vô hạn (.repeat()),
        validation_steps cần truyền vào model.fit() để đánh giá đúng số batch.
    """
    _X_img, _X_tab, _y = X_img, X_tab, y
    n_val = len(_y)
    validation_steps = int(np.ceil(n_val / batch_size))

    def generator():
        """Generator lặp vô tận để tương thích với .repeat() ngầm định."""
        while True:
            for i in range(n_val):
                yield (
                    {
                        "image_input":   _X_img[i].astype(np.float32),
                        "tabular_input": _X_tab[i].astype(np.float32),
                    },
                    np.float32(_y[i]),
                )

    ds = (
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
    return ds, validation_steps


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
def setup_mlflow_experiment(experiment_name):
    from mlflow.exceptions import MlflowException
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException as e:
        if "deleted" in str(e).lower():
            print(f"  [!] Experiment '{experiment_name}' đã bị xóa (soft-deleted). Đang khôi phục...")
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp:
                client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(experiment_name)
            else:
                raise e
        else:
            raise e


def main():
    try:
        MetricsServer.start()
    except NameError:
        pass

    print("=" * 60)
    print("BƯỚC 5: Two-Phase Training  [memory-safe build]")
    print(f"  Bucket: s3://{S3_OUTPUT_BUCKET}/preprocessed/")
    print("=" * 60)

    # ── Skip nếu cả 2 model đã được train và upload lên S3 ──────────────
    KEY_PHASE1 = "preprocessed/best_model_phase1.h5"
    KEY_PHASE2 = "preprocessed/best_model_isic2024.h5"
    if s3_key_exists(KEY_PHASE1, bucket=S3_OUTPUT_BUCKET) and \
       s3_key_exists(KEY_PHASE2, bucket=S3_OUTPUT_BUCKET):
        print(f"\n[SKIP] Cả 2 model đã tồn tại trên S3:")
        print(f"  ✓ s3://{S3_OUTPUT_BUCKET}/{KEY_PHASE1}")
        print(f"  ✓ s3://{S3_OUTPUT_BUCKET}/{KEY_PHASE2}")
        print("  → Tiến hành tải model từ S3 để đăng ký vào MLflow Registry...")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        setup_mlflow_experiment(MLFLOW_EXPERIMENT)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            local_model_path = tmp.name

        download_file(KEY_PHASE2, local_model_path, bucket=S3_OUTPUT_BUCKET)

        from tensorflow.keras.models import load_model
        model = load_model(local_model_path, compile=False)

        with mlflow.start_run(run_name="register-existing-model") as run:
            run_id = run.info.run_id
            print("\nĐăng ký model vào MLflow Model Registry (bypass create_logged_model)...")
            artifact_path = "model"

            with tempfile.TemporaryDirectory() as td:
                local_model_dir = os.path.join(td, "model_dir")
                mlflow.tensorflow.save_model(model, path=local_model_dir)
                
                s3_base_key = f"mlflow_artifacts/{run_id}/{artifact_path}"
                for root, dirs, files in os.walk(local_model_dir):
                    for file in files:
                        local_path = os.path.join(root, file)
                        rel_path = os.path.relpath(local_path, local_model_dir)
                        s3_key = f"{s3_base_key}/{rel_path}".replace("\\", "/")
                        upload_file(local_path, s3_key, bucket=S3_OUTPUT_BUCKET)

            model_uri = f"s3://{S3_OUTPUT_BUCKET}/{s3_base_key}"
            mv = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
            print(f"  ✓ Registered: {MLFLOW_MODEL_NAME} v{mv.version}")

        os.unlink(local_model_path)
        sys.exit(0)

    # ── Khởi tạo MLflow ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    setup_mlflow_experiment(MLFLOW_EXPERIMENT)

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
    train_ds, steps_per_epoch = build_train_dataset(
        X_img_train, X_tab_train, y_train, tabular_dim,
        batch_size=BATCH_SIZE, oversample_ratio=OVERSAMPLE_RATIO,
    )
    val_ds, validation_steps = build_val_dataset(
        X_img_val, X_tab_val, y_val, tabular_dim, batch_size=BATCH_SIZE,
    )

    # ── Mở 1 MLflow run cho toàn bộ quá trình training ──────────────────
    with mlflow.start_run(run_name="two-phase-training") as run:
        run_id = run.info.run_id
        print(f"\n  MLflow Run ID: {run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "phase1_epochs":    PHASE1_EPOCHS,
            "phase2_epochs":    PHASE2_EPOCHS,
            "batch_size":       BATCH_SIZE,
            "phase2_batch":     PHASE2_BATCH,
            "oversample_ratio": OVERSAMPLE_RATIO,
            "class_weight_mal": CLASS_W_MAL,
            "fine_tune_from":   FINE_TUNE_FROM,
            "phase1_lr":        PHASE1_LR,
            "phase2_lr":        PHASE2_LR,
            "image_size":       IMAGE_SIZE,
            "tabular_dim":      tabular_dim,
            "n_train":          len(y_train),
            "n_val":            len(y_val),
            "n_gpu":            len(gpus),
            "s3_bucket":        S3_OUTPUT_BUCKET,
        })

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
            MlflowEpochLogger(phase="phase1"),
            PrometheusEpochLogger(total_epochs=PHASE1_EPOCHS, phase="phase1"),
        ]

        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=PHASE1_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=cb1,
            class_weight=class_weight,
            verbose=1,
        )

        # Log best phase1 metrics
        best_p1_auc = max(h1.history.get("val_auc", [0.0]))
        mlflow.log_metric("phase1/best_val_auc", best_p1_auc)

        upload_file(phase1_local, KEY_PHASE1, bucket=S3_OUTPUT_BUCKET)
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
        train_ds_p2, steps_per_epoch_p2 = build_train_dataset(
            X_img_train, X_tab_train, y_train, tabular_dim,
            batch_size=PHASE2_BATCH, oversample_ratio=OVERSAMPLE_RATIO,
        )
        val_ds_p2, validation_steps_p2 = build_val_dataset(
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
            MlflowEpochLogger(phase="phase2"),
            PrometheusEpochLogger(total_epochs=PHASE2_EPOCHS, phase="phase2"),
        ]

        h2 = model.fit(
            train_ds_p2,
            validation_data=val_ds_p2,
            epochs=PHASE2_EPOCHS,
            steps_per_epoch=steps_per_epoch_p2,
            validation_steps=validation_steps_p2,
            callbacks=cb2,
            class_weight=class_weight,
            verbose=1,
        )

        # Log best phase2 metrics
        best_p2_auc     = max(h2.history.get("val_auc",       [0.0]))
        best_p2_recall  = max(h2.history.get("val_recall",    [0.0]))
        best_p2_prec    = max(h2.history.get("val_precision",  [0.0]))
        mlflow.log_metric("phase2/best_val_auc",       best_p2_auc)
        mlflow.log_metric("phase2/best_val_recall",    best_p2_recall)
        mlflow.log_metric("phase2/best_val_precision", best_p2_prec)

        # Upload model tốt nhất (phase 2) lên S3
        upload_file(best_local, KEY_PHASE2, bucket=S3_OUTPUT_BUCKET)

        # ── Đăng ký model vào MLflow Model Registry ──────────────────
        print("\n[5/5] Đăng ký model vào MLflow Model Registry (bypass create_logged_model)...")
        artifact_path = "model"

        with tempfile.TemporaryDirectory(dir=TMP_DIR) as td:
            local_model_dir = os.path.join(td, "model_dir")
            mlflow.tensorflow.save_model(model, path=local_model_dir)
            
            s3_base_key = f"mlflow_artifacts/{run_id}/{artifact_path}"
            for root, dirs, files in os.walk(local_model_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, local_model_dir)
                    s3_key = f"{s3_base_key}/{rel_path}".replace("\\", "/")
                    upload_file(local_path, s3_key, bucket=S3_OUTPUT_BUCKET)

        model_uri = f"s3://{S3_OUTPUT_BUCKET}/{s3_base_key}"
        mv = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
        print(f"  ✓ Registered: {MLFLOW_MODEL_NAME} v{mv.version}")

        # Tag run với S3 path và best AUC để dễ tra cứu
        mlflow.set_tags({
            "s3_model_phase1": f"s3://{S3_OUTPUT_BUCKET}/{KEY_PHASE1}",
            "s3_model_phase2": f"s3://{S3_OUTPUT_BUCKET}/{KEY_PHASE2}",
            "best_val_auc":    str(round(best_p2_auc, 4)),
            "model_version":   str(mv.version),
        })

        os.unlink(best_local)

        # Lưu history lên S3
        save_pkl(
            {"phase1": h1.history, "phase2": h2.history},
            "preprocessed/training_history.pkl",
            bucket=S3_OUTPUT_BUCKET,
        )

    print(f"\n  Model → s3://{S3_OUTPUT_BUCKET}/{KEY_PHASE2}")
    print(f"  MLflow Run: {MLFLOW_TRACKING_URI}/#/experiments/{MLFLOW_EXPERIMENT}/runs/{run_id}")
    print(f"  Registry  : {MLFLOW_MODEL_NAME} v{mv.version}")
    print("\nBước 5 hoàn thành!")


if __name__ == "__main__":
    main()
