"""
build_mlp.py — Bước 4b: Định nghĩa Tabular Branch (MLP)

Đọc : s3://kltn-isic-2024-colab/preprocessed/encoders.pkl  (lấy tabular_dim)
Ghi :
  s3://kltn-isic-2024-colab/preprocessed/mlp_architecture.json
  s3://kltn-isic-2024-colab/preprocessed/mlp_meta.json
"""
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

from s3_utils import load_pkl, upload_bytes, S3_OUTPUT_BUCKET


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
    encoders     = load_pkl("preprocessed/encoders.pkl", bucket=S3_OUTPUT_BUCKET)
    feature_cols = encoders["feature_cols"]
    tabular_dim  = len(feature_cols)
    print(f"Tabular dimension: {tabular_dim} features")

    model = build_mlp_branch(tabular_dim)
    model.summary()

    upload_bytes(
        model.to_json().encode(),
        "preprocessed/mlp_architecture.json",
        bucket=S3_OUTPUT_BUCKET,
    )

    meta = {
        "tabular_dim":   tabular_dim,
        "feature_cols":  feature_cols,
        "output_dim":    32,
        "total_params":  model.count_params(),
    }
    upload_bytes(
        json.dumps(meta, indent=2).encode(),
        "preprocessed/mlp_meta.json",
        bucket=S3_OUTPUT_BUCKET,
    )

    print(f"\nLưu → s3://{S3_OUTPUT_BUCKET}/preprocessed/mlp_architecture.json")
    print(f"Lưu → s3://{S3_OUTPUT_BUCKET}/preprocessed/mlp_meta.json")
    print("\nBước 4b hoàn thành!")


if __name__ == "__main__":
    main()
