"""
Multimodal CNN model for Skin Cancer Classification.
Combines image branch (CNN) with tabular branch (Dense) and fuses them.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Concatenate,
)


def build_image_branch(image_shape: tuple[int, int, int]) -> tuple:
    """
    Lightweight CNN branch for processing skin lesion images.

    Architecture: Conv → BN → Pool → Conv → BN → Pool → GAP
    """
    image_input = Input(shape=image_shape, name="image_input")

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    image_features = Dropout(0.5)(x)

    return image_input, image_features


def build_tabular_branch(tabular_shape: tuple[int]) -> tuple:
    """
    Dense branch for processing tabular / EHR features.
    """
    tabular_input = Input(shape=tabular_shape, name="tabular_input")

    x = Dense(64, activation="relu")(tabular_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation="relu")(x)
    tabular_features = Dropout(0.3)(x)

    return tabular_input, tabular_features


def build_multimodal_model(
    tabular_shape: tuple[int],
    image_shape: tuple[int, int, int],
    num_classes: int = 3,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Fuse image and tabular branches and return a compiled Keras model.

    Parameters
    ----------
    tabular_shape  : e.g. (4,)
    image_shape    : e.g. (224, 224, 3)
    num_classes    : number of diagnostic classes
    learning_rate  : Adam LR

    Returns
    -------
    model : compiled tf.keras.Model
    """
    image_input, image_features = build_image_branch(image_shape)
    tabular_input, tabular_features = build_tabular_branch(tabular_shape)

    # Fusion
    fused = Concatenate(name="fusion")([image_features, tabular_features])
    x = Dense(128, activation="relu")(fused)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(
        inputs=[image_input, tabular_input],
        outputs=output,
        name="SkinCancerMultimodal",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model: Model) -> str:
    """Return the model summary as a string."""
    lines: list[str] = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)
