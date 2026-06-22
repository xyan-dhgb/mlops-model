"""
build_efficientnetB3.py — Bước 4a: Định nghĩa Image Branch

Ghi:
  DVC/Local (preprocessed/efficientnetB3_architecture.json)
  DVC/Local (preprocessed/efficientnetB3_meta.json)
# Cũ: s3://kltn-isic-2024-colab/preprocessed/efficientnetB3_architecture.json
# Cũ: s3://kltn-isic-2024-colab/preprocessed/efficientnetB3_meta.json
  (weights không lưu — model được build lại khi train)
"""
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import s3_utils
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization,
)

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
os.makedirs(os.path.join(DATA_DIR, "preprocessed"), exist_ok=True)

IMAGE_SIZE  = int(os.environ.get("IMAGE_SIZE", "224"))
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal Loss: FL(p) = -α(1-p)^γ log(p)"""
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce    = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_t * focal_weight * bce)
    focal_loss_fn.__name__ = "focal_loss"
    return focal_loss_fn


def build_image_branch(image_shape=IMAGE_SHAPE):
    """
    EfficientNetB3(ImageNet, frozen) → GAP → BN
    → Dense(256,relu) → Dropout(0.4)
    → Dense(128,relu) → Dropout(0.3)
    Output dim: 128
    """
    image_input = Input(shape=image_shape, name="image_input")
    backbone = EfficientNetB3(
        include_top=False, weights="imagenet",
        input_tensor=image_input, pooling=None,
    )
    backbone.trainable = False

    x = backbone.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn_image")(x)
    x = Dense(256, activation="relu", name="dense_img_256")(x)
    x = Dropout(0.4, name="drop_img_1")(x)
    x = Dense(128, activation="relu", name="dense_img_128")(x)
    x = Dropout(0.3, name="drop_img_2")(x)

    model = Model(inputs=image_input, outputs=x, name="efficientnetB3_branch")
    return model, backbone


def main():
    print(f"TensorFlow: {tf.__version__}")
    print(f"Image shape: {IMAGE_SHAPE}")

    model, backbone = build_image_branch()
    model.summary()

    # Lưu architecture JSON
    arch_json = model.to_json()
    arch_path = os.path.join(DATA_DIR, "preprocessed/efficientnetB3_architecture.json")
    with open(arch_path, "w", encoding="utf-8") as f:
        f.write(arch_json)

    meta = {
        "image_shape":         list(IMAGE_SHAPE),
        "backbone":            "EfficientNetB3",
        "backbone_layers":     len(backbone.layers),
        "fine_tune_from_layer": 300,
        "conv_last_layer":     "top_conv",
        "output_dim":          128,
        "total_params":        model.count_params(),
    }
    meta_path = os.path.join(DATA_DIR, "preprocessed/efficientnetB3_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLưu → {arch_path}")
    print(f"Lưu → {meta_path}")

    print("\nĐồng bộ file lên S3 Output Bucket (thay thế DVC)...")
    s3_utils.upload_file(arch_path, "preprocessed/efficientnetB3_architecture.json")
    s3_utils.upload_file(meta_path, "preprocessed/efficientnetB3_meta.json")

    print("\nBước 4a hoàn thành!")


if __name__ == "__main__":
    main()
