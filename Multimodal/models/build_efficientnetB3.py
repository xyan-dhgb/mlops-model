"""
build_efficientnetB3.py — Bước 4a: Định nghĩa Image Branch (EfficientNetB3)
Kiến trúc: EfficientNetB3(ImageNet) → GAP → BN → Dense(256) → Dropout(0.4)
                                             → Dense(128) → Dropout(0.3)
Đầu ra  : /data/model/efficientnetB3_branch.h5
           /data/model/efficientnetB3_meta.json
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization,
)

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/data/processed")
MODEL_DIR     = os.environ.get("MODEL_DIR", "/data/model")
IMAGE_SIZE    = int(os.environ.get("IMAGE_SIZE", "224"))

os.makedirs(MODEL_DIR, exist_ok=True)

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
    Image Branch: EfficientNetB3 pretrained ImageNet
    Trả về (model_image_branch, backbone) để dùng lại ở bước train.
    """
    image_input = Input(shape=image_shape, name="image_input")

    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        pooling=None,
    )
    backbone.trainable = False  # Phase 1: đóng băng

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

    # Lưu weights khởi tạo
    weights_path = os.path.join(MODEL_DIR, "efficientnetB3_branch.h5")
    model.save_weights(weights_path)
    print(f"Lưu weights → {weights_path}")

    # Lưu architecture JSON
    arch_path = os.path.join(MODEL_DIR, "efficientnetB3_architecture.json")
    with open(arch_path, "w") as f:
        f.write(model.to_json())
    print(f"Lưu architecture → {arch_path}")

    # Metadata
    meta = {
        "image_shape": list(IMAGE_SHAPE),
        "backbone": "EfficientNetB3",
        "backbone_layers": len(backbone.layers),
        "fine_tune_from_layer": 300,
        "conv_last_layer": "top_conv",
        "output_dim": 128,
        "total_params": model.count_params(),
        "trainable_params_phase1": sum(
            np.prod(v.shape) for v in model.trainable_weights
        ),
    }
    meta_path = os.path.join(MODEL_DIR, "efficientnetB3_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Lưu metadata → {meta_path}")

    model.summary()
    print("\nBuild EfficientNetB3 branch hoàn thành!")


if __name__ == "__main__":
    main()
