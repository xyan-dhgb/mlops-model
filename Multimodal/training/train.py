"""
train.py — Bước 5: Huấn luyện 2 giai đoạn Multimodal

Đọc từ S3:
  s3://kltn-isic-2024-colab/splits/train/X_tab_train.npy
  s3://kltn-isic-2024-colab/splits/train/X_img_train.npy
  s3://kltn-isic-2024-colab/splits/train/y_train.npy
  s3://kltn-isic-2024-colab/splits/val/  (tương tự)
  s3://kltn-isic-2024-colab/preprocessed/encoders.pkl

Ghi lên S3:
  s3://kltn-isic-2024-colab/preprocessed/best_model_phase1.h5
  s3://kltn-isic-2024-colab/preprocessed/best_model_isic2024.h5
  s3://kltn-isic-2024-colab/preprocessed/training_history.pkl
  s3://kltn-isic-2024-colab/preprocessed/model_architecture.json

Khớp notebook cell 36 (train_model):
  Phase 1: backbone frozen, LR=1e-3, EarlyStopping(val_auc, patience=5)
  Phase 2: unfreeze ≥ layer 300, LR=1e-4, EarlyStopping(val_auc, patience=7)
  class_weight: (n_neg/n_pos)×1.2  [giảm từ 1.5 → 1.2 như notebook]
"""
import io
import json
import os
import pickle
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

from augment import oversample_malignant
from s3_utils import (
    load_npy, load_pkl, save_pkl, upload_bytes,
    save_keras_model,
    S3_OUTPUT_BUCKET,
)

# ── Hyperparameters ──────────────────────────────────────────────────────
PHASE1_EPOCHS    = int(os.environ.get("PHASE1_EPOCHS", "20"))
PHASE2_EPOCHS    = int(os.environ.get("PHASE2_EPOCHS", "10"))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "32"))
OVERSAMPLE_RATIO = float(os.environ.get("OVERSAMPLE_RATIO", "0.25"))
CLASS_W_MAL      = float(os.environ.get("CLASS_WEIGHT_MAL", "1.2"))
FINE_TUNE_FROM   = int(os.environ.get("FINE_TUNE_FROM_LAYER", "300"))
PHASE1_LR        = float(os.environ.get("PHASE1_LR", "1e-3"))
PHASE2_LR        = float(os.environ.get("PHASE2_LR", "1e-4"))
IMAGE_SIZE       = int(os.environ.get("IMAGE_SIZE", "224"))
IMAGE_SHAPE      = (IMAGE_SIZE, IMAGE_SIZE, 3)


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    fn.__name__ = "focal_loss"
    return fn


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


def main():
    print("=" * 60)
    print("BƯỚC 5: Two-Phase Training")
    print(f"  GPU: {tf.config.list_physical_devices('GPU')}")
    print(f"  Bucket: s3://{S3_OUTPUT_BUCKET}/preprocessed/")
    print("=" * 60)

    # ── Load splits từ S3 ────────────────────────────────────────────────
    print("\nĐọc splits từ S3...")
    X_tab_train = load_npy("splits/train/X_tab_train.npy", bucket=S3_OUTPUT_BUCKET)
    X_img_train = load_npy("splits/train/X_img_train.npy", bucket=S3_OUTPUT_BUCKET)
    y_train     = load_npy("splits/train/y_train.npy",     bucket=S3_OUTPUT_BUCKET)
    X_tab_val   = load_npy("splits/val/X_tab_val.npy",     bucket=S3_OUTPUT_BUCKET)
    X_img_val   = load_npy("splits/val/X_img_val.npy",     bucket=S3_OUTPUT_BUCKET)
    y_val       = load_npy("splits/val/y_val.npy",         bucket=S3_OUTPUT_BUCKET)

    encoders     = load_pkl("preprocessed/encoders.pkl", bucket=S3_OUTPUT_BUCKET)
    tabular_dim  = len(encoders["feature_cols"])
    print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | tabular_dim={tabular_dim}")

    # ── Oversampling Malignant (cell 36) ────────────────────────────────
    X_img_os, X_tab_os, y_os = oversample_malignant(
        X_img_train, X_tab_train, y_train,
        target_ratio=OVERSAMPLE_RATIO, strong_aug=True,
    )

    # Class weight ×1.2 (giảm từ 1.5 → 1.2 như notebook)
    n_neg = int((y_os == 0).sum())
    n_pos = int((y_os == 1).sum())
    class_weight = {0: 1.0, 1: (n_neg / n_pos) * CLASS_W_MAL}
    print(f"\nClass weights: {class_weight}")

    # ── Build model ──────────────────────────────────────────────────────
    model, backbone = build_multimodal_model(tabular_dim, freeze_backbone=True)
    model = compile_model(model, PHASE1_LR)

    # Lưu architecture
    upload_bytes(
        model.to_json().encode(),
        "preprocessed/model_architecture.json",
        bucket=S3_OUTPUT_BUCKET,
    )

    # ── PHASE 1 (cell 36: frozen, patience=5) ───────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen backbone")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp1:
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
        {"image_input": X_img_os, "tabular_input": X_tab_os}, y_os,
        validation_data=({"image_input": X_img_val, "tabular_input": X_tab_val}, y_val),
        epochs=PHASE1_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cb1, class_weight=class_weight, verbose=1,
    )

    # Upload phase1 model lên S3
    from s3_utils import upload_file
    upload_file(phase1_local, "preprocessed/best_model_phase1.h5", bucket=S3_OUTPUT_BUCKET)
    os.unlink(phase1_local)

    # ── PHASE 2 (cell 36: unfreeze ≥ layer 300, patience=7) ─────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-tune EfficientNetB3 từ layer {FINE_TUNE_FROM}")
    print("=" * 60)

    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_FROM]:
        layer.trainable = False

    print(f"Layers mở đóng băng: "
          f"{sum(1 for l in backbone.layers if l.trainable)}/{len(backbone.layers)}")

    model = compile_model(model, PHASE2_LR)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp2:
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
        {"image_input": X_img_os, "tabular_input": X_tab_os}, y_os,
        validation_data=({"image_input": X_img_val, "tabular_input": X_tab_val}, y_val),
        epochs=PHASE2_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cb2, class_weight=class_weight, verbose=1,
    )

    # Upload best model
    upload_file(best_local, "preprocessed/best_model_isic2024.h5", bucket=S3_OUTPUT_BUCKET)
    os.unlink(best_local)

    # Lưu history
    save_pkl(
        {"phase1": h1.history, "phase2": h2.history},
        "preprocessed/training_history.pkl",
        bucket=S3_OUTPUT_BUCKET,
    )

    print(f"\nModel → s3://{S3_OUTPUT_BUCKET}/preprocessed/best_model_isic2024.h5")
    print("\nBước 5 hoàn thành!")


if __name__ == "__main__":
    main()
