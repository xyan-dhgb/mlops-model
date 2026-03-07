#!/usr/bin/env python3
"""
Entrypoint script for training the Skin Cancer Multimodal model.

Environment variables
---------------------
DATA_DIR        : path to dataset root (must contain metadata.csv + all_images/)
MODEL_OUTPUT    : where to save the final .h5 model  (default: /app/output/model.h5)
EPOCHS          : training epochs                     (default: 20)
BATCH_SIZE      : batch size                          (default: 32)
IMAGE_SIZE      : comma-separated H,W e.g. 224,224   (default: 224,224)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import (
    load_csv_data,
    preprocess_csv_data,
    prepare_multimodal_data,
    split_dataset,
)
from src.model import build_multimodal_model, get_model_summary
from src.train import train_model, evaluate_model


# ── Config from environment ─────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "/data")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "/app/output/model.h5")
EPOCHS = int(os.getenv("EPOCHS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
_img_size = os.getenv("IMAGE_SIZE", "224,224").split(",")
IMAGE_SIZE = (int(_img_size[0]), int(_img_size[1]))

CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "all_images")
CHECKPOINT_DIR = os.path.join(os.path.dirname(MODEL_OUTPUT), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(MODEL_OUTPUT), "logs")
METRICS_PATH = os.path.join(os.path.dirname(MODEL_OUTPUT), "metrics.json")


def main():
    print("=" * 60)
    print("  Skin Cancer Multimodal Classification – Training")
    print("=" * 60)

    # 1. Load & preprocess metadata
    print("\n[1/5] Loading and preprocessing CSV data …")
    df = load_csv_data(CSV_PATH)
    df, report = preprocess_csv_data(df)
    print(f"      Shape: {report['final_shape']}  |  Missing: {report['missing_after']}")

    # 2. Build multimodal arrays
    print("\n[2/5] Preparing multimodal data (images + tabular) …")
    X_tabular, X_image, y, label_encoder = prepare_multimodal_data(
        df, IMAGE_DIR, target_size=IMAGE_SIZE
    )
    print(f"      Tabular: {X_tabular.shape}  |  Image: {X_image.shape}  |  Labels: {len(y)}")

    # 3. Split
    print("\n[3/5] Splitting dataset …")
    splits = split_dataset(X_tabular, X_image, y)
    print(
        f"      Train={len(splits['y_train'])}  Val={len(splits['y_val'])}  "
        f"Test={len(splits['y_test'])}"
    )

    # 4. Build & train
    print("\n[4/5] Building and training model …")
    num_classes = len(np.unique(y))
    model = build_multimodal_model(
        tabular_shape=splits["X_tab_train"].shape[1:],
        image_shape=splits["X_img_train"].shape[1:],
        num_classes=num_classes,
    )
    print(get_model_summary(model))

    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    train_model(
        model,
        splits["X_tab_train"], splits["X_img_train"], splits["y_train"],
        splits["X_tab_val"],   splits["X_img_val"],   splits["y_val"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
    )

    # 5. Evaluate & save
    print("\n[5/5] Evaluating on test set and saving model …")
    metrics = evaluate_model(
        model,
        splits["X_tab_test"], splits["X_img_test"], splits["y_test"],
    )
    print(f"      Test Loss: {metrics['loss']:.4f}  |  Test Accuracy: {metrics['accuracy']:.4f}")

    model.save(MODEL_OUTPUT)
    print(f"\n✅  Model saved to: {MODEL_OUTPUT}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅  Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
