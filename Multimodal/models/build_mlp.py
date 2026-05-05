"""
build_mlp.py — Bước 4b: Định nghĩa Tabular Branch (MLP)
Kiến trúc: Input(tabular_dim) → Dense(128) → BN → Dropout(0.3)
                               → Dense(64)  → Dropout(0.2)
                               → Dense(32)
Đầu vào : /data/processed/encoders.pkl  (để lấy tabular_dim)
Đầu ra  : /data/model/mlp_branch.h5
           /data/model/mlp_meta.json
           /data/model/mlp_architecture.json
"""
import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
)

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/data/processed")
MODEL_DIR     = os.environ.get("MODEL_DIR", "/data/model")

os.makedirs(MODEL_DIR, exist_ok=True)

ENCODERS_PATH = os.path.join(PROCESSED_DIR, "encoders.pkl")


def build_mlp_branch(tabular_dim: int) -> Model:
    """
    Tabular Branch (MLP):
      Dense(128, relu) → BN → Dropout(0.3)
      → Dense(64, relu) → Dropout(0.2)
      → Dense(32, relu)
    Output dim = 32 (sẽ được Concatenate với image branch)
    """
    tabular_input = Input(shape=(tabular_dim,), name="tabular_input")

    x = Dense(128, activation="relu", name="mlp_dense_128")(tabular_input)
    x = BatchNormalization(name="mlp_bn")(x)
    x = Dropout(0.3, name="mlp_drop_1")(x)
    x = Dense(64, activation="relu", name="mlp_dense_64")(x)
    x = Dropout(0.2, name="mlp_drop_2")(x)
    x = Dense(32, activation="relu", name="mlp_dense_32")(x)

    model = Model(inputs=tabular_input, outputs=x, name="mlp_branch")
    return model


def main():
    # Lấy tabular_dim từ encoders đã lưu
    if not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError(f"Không tìm thấy {ENCODERS_PATH}. "
                                "Chạy preprocess_csv trước!")

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    feature_cols = encoders.get("feature_cols", [])
    tabular_dim  = len(feature_cols)
    print(f"Tabular dimension: {tabular_dim} features")
    print(f"Các feature: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    model = build_mlp_branch(tabular_dim)

    # Lưu weights khởi tạo
    weights_path = os.path.join(MODEL_DIR, "mlp_branch.h5")
    model.save_weights(weights_path)
    print(f"Lưu weights → {weights_path}")

    # Lưu architecture JSON
    arch_path = os.path.join(MODEL_DIR, "mlp_architecture.json")
    with open(arch_path, "w") as f:
        f.write(model.to_json())
    print(f"Lưu architecture → {arch_path}")

    # Metadata
    meta = {
        "tabular_dim": tabular_dim,
        "feature_cols": feature_cols,
        "output_dim": 32,
        "layers": ["Dense(128)", "BN", "Dropout(0.3)",
                   "Dense(64)", "Dropout(0.2)", "Dense(32)"],
        "total_params": model.count_params(),
    }
    meta_path = os.path.join(MODEL_DIR, "mlp_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Lưu metadata → {meta_path}")

    model.summary()
    print("\nBuild MLP branch hoàn thành!")


if __name__ == "__main__":
    main()
