"""
Training utilities for the Skin Cancer Multimodal model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.models import Model


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────
def get_callbacks(
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    patience: int = 5,
) -> list:
    """Standard callbacks: EarlyStopping, ModelCheckpoint, ReduceLR, TensorBoard."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_model(
    model: Model,
    X_tab_train: np.ndarray,
    X_img_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_val: np.ndarray,
    X_img_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
) -> tf.keras.callbacks.History:
    """
    Train the multimodal model.

    Returns
    -------
    history : Keras History object
    """
    callbacks = get_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    history = model.fit(
        x={"image_input": X_img_train, "tabular_input": X_tab_train},
        y=y_train,
        validation_data=(
            {"image_input": X_img_val, "tabular_input": X_tab_val},
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate_model(
    model: Model,
    X_tab_test: np.ndarray,
    X_img_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate on the test set.

    Returns
    -------
    metrics : dict with keys 'loss' and 'accuracy'
    """
    loss, accuracy = model.evaluate(
        x={"image_input": X_img_test, "tabular_input": X_tab_test},
        y=y_test,
        verbose=0,
    )
    return {"loss": float(loss), "accuracy": float(accuracy)}


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(
    model: Model,
    X_tab: np.ndarray,
    X_img: np.ndarray,
) -> np.ndarray:
    """
    Run inference and return class probabilities.
    """
    return model.predict(
        {"image_input": X_img, "tabular_input": X_tab},
        verbose=0,
    )
