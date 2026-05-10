"""
build_efficientnetB3.py — Bước 4a: Định nghĩa Image Branch

Ghi:
  s3://kltn-isic-2024-colab/preprocessed/efficientnetB3_architecture.json
  s3://kltn-isic-2024-colab/preprocessed/efficientnetB3_meta.json
  (weights không lưu — model được build lại khi train)
"""
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization,
)

from s3_utils import upload_bytes, S3_OUTPUT_BUCKET

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

    # Lưu architecture JSON lên S3
    arch_json = model.to_json()
    upload_bytes(
        arch_json.encode(),
        "preprocessed/efficientnetB3_architecture.json",
        bucket=S3_OUTPUT_BUCKET,
    )

    meta = {
        "image_shape":         list(IMAGE_SHAPE),
        "backbone":            "EfficientNetB3",
        "backbone_layers":     len(backbone.layers),
        "fine_tune_from_layer": 300,
        "conv_last_layer":     "top_conv",
        "output_dim":          128,
        "total_params":        model.count_params(),
    }
    upload_bytes(
        json.dumps(meta, indent=2).encode(),
        "preprocessed/efficientnetB3_meta.json",
        bucket=S3_OUTPUT_BUCKET,
    )

    print(f"\nLưu → s3://{S3_OUTPUT_BUCKET}/preprocessed/efficientnetB3_architecture.json")
    print(f"Lưu → s3://{S3_OUTPUT_BUCKET}/preprocessed/efficientnetB3_meta.json")
    print("\nBước 4a hoàn thành!")


if __name__ == "__main__":
    main()
