"""
Multimodal Model for ISIC 2024 Skin Lesion Classification
Binary classification: Benign (0) vs Malignant (1)
Architecture: CNN branch (image) + MLP branch (tabular) -> Concatenation -> Output
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten,
    Concatenate, Conv2D, MaxPooling2D,
    BatchNormalization, GlobalAveragePooling2D
)


def build_multimodal_model(
    tabular_shape: tuple,
    image_shape: tuple = (224, 224, 3),
    num_classes: int = 2
) -> Model:
    """
    Multimodal model combining image (CNN) and tabular (MLP) branches.

    Args:
        tabular_shape: Shape of tabular input, e.g. (50,)
        image_shape:   Shape of image input, e.g. (224, 224, 3)
        num_classes:   Number of output classes (2 for ISIC 2024 binary)

    Returns:
        Compiled Keras Model
    """
    # ── Image branch (CNN) ──────────────────────────────────────────────────
    image_input = Input(shape=image_shape, name="image_input")

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)

    # ── Tabular branch (MLP) ────────────────────────────────────────────────
    tabular_input = Input(shape=tabular_shape, name="tabular_input")

    y = Dense(128, activation="relu")(tabular_input)
    y = Dropout(0.3)(y)
    y = Dense(64, activation="relu")(y)
    y = Dropout(0.2)(y)
    y = Dense(32, activation="relu")(y)

    # ── Fusion ──────────────────────────────────────────────────────────────
    combined = Concatenate()([x, y])
    z = Dense(128, activation="relu")(combined)
    z = Dropout(0.4)(z)
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.3)(z)

    # ── Output ──────────────────────────────────────────────────────────────
    if num_classes == 2:
        output = Dense(1, activation="sigmoid", name="output")(z)
        loss = "binary_crossentropy"
    else:
        output = Dense(num_classes, activation="softmax", name="output")(z)
        loss = "sparse_categorical_crossentropy"

    model = Model(
        inputs=[image_input, tabular_input],
        outputs=output,
        name="MultimodalISIC2024"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


if __name__ == "__main__":
    # Quick smoke-test
    model = build_multimodal_model(tabular_shape=(50,), image_shape=(224, 224, 3))
    model.summary()
