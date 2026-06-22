"""
build_mlp.py — Bước 4b: Định nghĩa Tabular Branch (MLP)

Đọc : DVC/Local (preprocessed/encoders.pkl)  (lấy tabular_dim)
# Cũ: s3://kltn-isic-2024-colab/preprocessed/encoders.pkl
Ghi :
  DVC/Local (preprocessed/mlp_architecture.json)
  DVC/Local (preprocessed/mlp_meta.json)
# Cũ: s3://kltn-isic-2024-colab/preprocessed/mlp_architecture.json
# Cũ: s3://kltn-isic-2024-colab/preprocessed/mlp_meta.json
"""
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import s3_utils

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
os.makedirs(os.path.join(DATA_DIR, "preprocessed"), exist_ok=True)


def build_mlp_branch(tabular_dim: int) -> Model:
    """
    MLP Branch (khớp notebook cell 32):
      Dense(128,relu) → BN → Dropout(0.3)
      → Dense(64,relu) → Dropout(0.2)
      → Dense(32,relu)
    Output dim: 32
    """
    tab_input = Input(shape=(tabular_dim,), name="tabular_input")
    x = Dense(128, activation="relu", name="mlp_dense_128")(tab_input)
    x = BatchNormalization(name="mlp_bn")(x)
    x = Dropout(0.3, name="mlp_drop_1")(x)
    x = Dense(64,  activation="relu", name="mlp_dense_64")(x)
    x = Dropout(0.2, name="mlp_drop_2")(x)
    x = Dense(32,  activation="relu", name="mlp_dense_32")(x)

    model = Model(inputs=tab_input, outputs=x, name="mlp_branch")
    return model


def main():
    with open(os.path.join(DATA_DIR, "preprocessed/encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    feature_cols = encoders["feature_cols"]
    tabular_dim  = len(feature_cols)
    print(f"Tabular dimension: {tabular_dim} features")

    model = build_mlp_branch(tabular_dim)
    model.summary()

    arch_path = os.path.join(DATA_DIR, "preprocessed/mlp_architecture.json")
    with open(arch_path, "w", encoding="utf-8") as f:
        f.write(model.to_json())

    meta = {
        "tabular_dim":   tabular_dim,
        "feature_cols":  feature_cols,
        "output_dim":    32,
        "total_params":  model.count_params(),
    }
    meta_path = os.path.join(DATA_DIR, "preprocessed/mlp_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLưu → {arch_path}")
    print(f"Lưu → {meta_path}")

    print("\nĐồng bộ file lên S3 Output Bucket (thay thế DVC)...")
    s3_utils.upload_file(arch_path, "preprocessed/mlp_architecture.json")
    s3_utils.upload_file(meta_path, "preprocessed/mlp_meta.json")

    print("\nBước 4b hoàn thành!")


if __name__ == "__main__":
    main()
