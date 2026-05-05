"""
train.py — Bước 5: Huấn luyện 2 giai đoạn (EfficientNetB3 + MLP Multimodal)
Phase 1: Backbone frozen  | LR=1e-3 | 20 epochs | EarlyStopping(val_auc, patience=5)
Phase 2: Unfreeze ≥ 300   | LR=1e-4 | 10 epochs | EarlyStopping(val_auc, patience=7)

Đầu vào:
  /data/processed/images/<isic_id>.png
  /data/processed/tabular_processed.pkl
  /data/processed/encoders.pkl
  /data/splits/{train,val}_idx.npy
  /data/model/efficientnetB3_meta.json
  /data/model/mlp_meta.json

Đầu ra:
  /data/output/best_model_phase1.h5
  /data/output/best_model_isic2024.h5
  /data/output/training_history.pkl
  /data/output/model_architecture.json
  /data/output/model_summary.txt
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate,
    GlobalAveragePooling2D, BatchNormalization,
)
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from PIL import Image
from tqdm import tqdm

from augment import oversample_malignant

# ── Hyperparameters từ biến môi trường ──────────────────────────────────
PROCESSED_DIR       = os.environ.get("PROCESSED_DIR", "/data/processed")
SPLITS_DIR          = os.environ.get("SPLITS_DIR", "/data/splits")
MODEL_DIR           = os.environ.get("MODEL_DIR", "/data/model")
OUTPUT_DIR          = os.environ.get("OUTPUT_DIR", "/data/output")
PHASE1_EPOCHS       = int(os.environ.get("PHASE1_EPOCHS", "20"))
PHASE2_EPOCHS       = int(os.environ.get("PHASE2_EPOCHS", "10"))
BATCH_SIZE          = int(os.environ.get("BATCH_SIZE", "32"))
OVERSAMPLE_RATIO    = float(os.environ.get("OVERSAMPLE_RATIO", "0.25"))
CLASS_WEIGHT_MAL    = float(os.environ.get("CLASS_WEIGHT_MAL", "1.2"))
FINE_TUNE_FROM      = int(os.environ.get("FINE_TUNE_FROM_LAYER", "300"))
PHASE1_LR           = float(os.environ.get("PHASE1_LR", "1e-3"))
PHASE2_LR           = float(os.environ.get("PHASE2_LR", "1e-4"))
IMAGE_SIZE          = int(os.environ.get("IMAGE_SIZE", "224"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_DIR  = os.path.join(PROCESSED_DIR, "images")
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


# ── Focal Loss ───────────────────────────────────────────────────────────
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce    = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    focal_loss_fn.__name__ = "focal_loss"
    return focal_loss_fn


# ── Build full multimodal model ──────────────────────────────────────────
def build_multimodal_model(tabular_dim: int,
                            freeze_backbone: bool = True) -> tuple:
    image_input  = Input(shape=IMAGE_SHAPE, name="image_input")
    backbone     = EfficientNetB3(
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

    tabular_input = Input(shape=(tabular_dim,), name="tabular_input")
    y_tab = Dense(128, activation="relu")(tabular_input)
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
    model  = Model(inputs=[image_input, tabular_input], outputs=output)
    return model, backbone


def compile_model(model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pauc", num_thresholds=1000,
                                  summation_method="interpolation", curve="ROC"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


# ── Tải dữ liệu ─────────────────────────────────────────────────────────
def load_split(df: pd.DataFrame,
               idx: np.ndarray,
               feature_cols: list,
               split_name: str = "") -> tuple:
    """Tải ảnh PNG + tabular cho một split, trả về (X_img, X_tab, y)."""
    records = df.iloc[idx].reset_index(drop=True)
    X_img, X_tab, y_list = [], [], []

    print(f"Đang tải {split_name} ({len(records)} mẫu)...")
    for _, row in tqdm(records.iterrows(), total=len(records)):
        isic_id = row.get("isic_id", row.get("isic_id", None))
        img_path = os.path.join(IMAGE_DIR, f"{isic_id}.png")
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        X_img.append(img)
        X_tab.append(row[feature_cols].values.astype(np.float32))
        y_list.append(int(row["target"]))

    return (np.array(X_img, dtype=np.float32),
            np.array(X_tab, dtype=np.float32),
            np.array(y_list, dtype=np.int32))


def main():
    print("=" * 60)
    print("ISIC 2024 — Two-Phase Multimodal Training")
    print(f"TF {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")
    print("=" * 60)

    # Load metadata
    df       = pd.read_pickle(os.path.join(PROCESSED_DIR, "tabular_processed.pkl"))
    encoders = pickle.load(open(os.path.join(PROCESSED_DIR, "encoders.pkl"), "rb"))
    feature_cols = encoders["feature_cols"]
    tabular_dim  = len(feature_cols)

    idx_train = np.load(os.path.join(SPLITS_DIR, "train_idx.npy"))
    idx_val   = np.load(os.path.join(SPLITS_DIR, "val_idx.npy"))

    X_img_train, X_tab_train, y_train = load_split(df, idx_train, feature_cols, "Train")
    X_img_val,   X_tab_val,   y_val   = load_split(df, idx_val,   feature_cols, "Val")

    # Oversampling Malignant
    X_img_os, X_tab_os, y_os = oversample_malignant(
        X_img_train, X_tab_train, y_train,
        target_ratio=OVERSAMPLE_RATIO, strong_aug=True,
    )

    # Class weights
    n_neg = int((y_os == 0).sum())
    n_pos = int((y_os == 1).sum())
    class_weight = {0: 1.0, 1: (n_neg / n_pos) * CLASS_WEIGHT_MAL}
    print(f"\nClass weights: {class_weight}")

    # Build model Phase 1 (backbone frozen)
    model, backbone = build_multimodal_model(tabular_dim, freeze_backbone=True)
    model = compile_model(model, PHASE1_LR)

    # Lưu architecture và summary
    arch_path = os.path.join(OUTPUT_DIR, "model_architecture.json")
    with open(arch_path, "w") as f:
        f.write(model.to_json())

    summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
    with open(summary_path, "w") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    # ── PHASE 1 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen backbone — Train Fusion Head + MLP")
    print("=" * 60)

    cb_phase1 = [
        EarlyStopping(monitor="val_auc", patience=5,
                      restore_best_weights=True, mode="max", verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_phase1.h5"),
                        monitor="val_auc", save_best_only=True,
                        mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    history1 = model.fit(
        {"image_input": X_img_os, "tabular_input": X_tab_os}, y_os,
        validation_data=({"image_input": X_img_val,
                          "tabular_input": X_tab_val}, y_val),
        epochs=PHASE1_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_phase1,
        class_weight=class_weight,
        verbose=1,
    )

    # ── PHASE 2 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-tune EfficientNetB3 từ layer {FINE_TUNE_FROM}")
    print("=" * 60)

    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_FROM]:
        layer.trainable = False

    n_unfreeze = sum(1 for l in backbone.layers if l.trainable)
    print(f"Mở đóng băng: {n_unfreeze}/{len(backbone.layers)} layers")

    model = compile_model(model, PHASE2_LR)

    cb_phase2 = [
        EarlyStopping(monitor="val_auc", patience=7,
                      restore_best_weights=True, mode="max", verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_isic2024.h5"),
                        monitor="val_auc", save_best_only=True,
                        mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    history2 = model.fit(
        {"image_input": X_img_os, "tabular_input": X_tab_os}, y_os,
        validation_data=({"image_input": X_img_val,
                          "tabular_input": X_tab_val}, y_val),
        epochs=PHASE2_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_phase2,
        class_weight=class_weight,
        verbose=1,
    )

    # Lưu training history
    hist_path = os.path.join(OUTPUT_DIR, "training_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump({"phase1": history1.history,
                     "phase2": history2.history}, f)
    print(f"\nLưu training history → {hist_path}")
    print("Training hoàn thành!")


if __name__ == "__main__":
    main()
