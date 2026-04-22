"""
model.py
========
ISIC 2024 – Multimodal Skin Lesion Classifier

Architecture (matches diagram):
  ┌─────────────────────────────────────────────────────────────────┐
  │  EfficientNetB3 Branch (Image)                                  │
  │    pretrained ImageNet, include_top=False                       │
  │    GlobalAveragePooling2D → [1536]                              │
  │    BatchNorm → Dense(256, ReLU, Dropout 0.4)                    │
  │    Dense(128, ReLU, Dropout 0.3)  → Output [128]               │
  ├─────────────────────────────────────────────────────────────────┤
  │  MLP Branch (Tabular / Metadata)                                │
  │    Input: 37 features                                           │
  │    Dense(128, ReLU, Dropout 0.3)                                │
  │    Dense(64, ReLU, Dropout 0.2)                                 │
  │    Dense(32, ReLU)  → Output [32]                               │
  ├─────────────────────────────────────────────────────────────────┤
  │  Fusion Head  Concatenate [128+32=160]                          │
  │    Dense(128, ReLU, Dropout 0.4)                                │
  │    Dense(64, ReLU, Dropout 0.3)                                 │
  │    Dense(1, sigmoid)  → P(Malignant) ∈ [0,1]                   │
  └─────────────────────────────────────────────────────────────────┘

Training strategy (Two-Phase):
  Phase 1 – Head Training   : backbone frozen, LR=1e-3, Adam, 15 epochs
  Phase 2 – Fine-tuning     : unfreeze from layer 300+, LR=1e-4, 15 epochs
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EFFICIENTNET_LAST_CONV = "top_activation"
FINE_TUNE_FROM = 300          # unfreeze EfficientNetB3 layers from index 300
IMAGE_SHAPE = (224, 224, 3)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75            # weight for Malignant class


# ---------------------------------------------------------------------------
# Custom Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
    """
    Binary Focal Loss.
    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)
    α=0.75 weights Malignant heavily (replaces standard BCE α=0.5).
    """
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * focal_weight * bce)

    focal_loss_fn.__name__ = f"focal_loss_g{gamma}_a{alpha}"
    return focal_loss_fn


# ---------------------------------------------------------------------------
# Partial AUC metric (pAUC @ TPR ≥ 80 %) – ISIC 2024 primary metric
# ---------------------------------------------------------------------------

def compute_pauc(y_true: np.ndarray, y_pred: np.ndarray, min_tpr: float = 0.80) -> float:
    """Normalised pAUC in [0, 1]."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(tpr[mask], fpr[mask]) / (1.0 - min_tpr))


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_multimodal_model(
    tabular_shape: Tuple[int, ...],
    image_shape: Tuple[int, int, int] = IMAGE_SHAPE,
    freeze_backbone: bool = True,
    fine_tune_from: int = FINE_TUNE_FROM,
    focal_gamma: float = FOCAL_GAMMA,
    focal_alpha: float = FOCAL_ALPHA,
    lr: float = 1e-3,
) -> Tuple[Model, EfficientNetB3]:
    """
    Build and compile the multimodal model.

    Returns
    -------
    model           : compiled Keras Model
    backbone        : EfficientNetB3 instance (needed for Phase-2 unfreezing)
    """
    # ── IMAGE BRANCH ──────────────────────────────────────────────────────
    image_input = Input(shape=image_shape, name="image_input")
    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        pooling=None,
    )

    if freeze_backbone:
        backbone.trainable = False
        logger.info("Phase 1: EfficientNetB3 backbone FROZEN (~%d layers)", len(backbone.layers))
    else:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
        unfrozen = sum(l.trainable for l in backbone.layers)
        logger.info("Phase 2: EfficientNetB3 – %d layers TRAINABLE (from layer %d)", unfrozen, fine_tune_from)

    x = backbone.output                        # (None, 7, 7, 1536)
    x = GlobalAveragePooling2D()(x)            # (None, 1536)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    img_out = Dropout(0.3)(x)                  # (None, 128)

    # ── TABULAR / MLP BRANCH ──────────────────────────────────────────────
    tabular_input = Input(shape=tabular_shape, name="tabular_input")
    t = Dense(128, activation="relu")(tabular_input)
    t = Dropout(0.3)(t)
    t = Dense(64, activation="relu")(t)
    t = Dropout(0.2)(t)
    tab_out = Dense(32, activation="relu")(t)  # (None, 32)

    # ── FUSION HEAD ───────────────────────────────────────────────────────
    fused = Concatenate()([img_out, tab_out])  # (None, 160)
    f = Dense(128, activation="relu")(fused)
    f = Dropout(0.4)(f)
    f = Dense(64, activation="relu")(f)
    f = Dropout(0.3)(f)
    output = Dense(1, activation="sigmoid", name="output")(f)

    model = Model(inputs=[image_input, tabular_input], outputs=output, name="multimodal_model")

    _compile_model(model, lr=lr, focal_gamma=focal_gamma, focal_alpha=focal_alpha)
    return model, backbone


def _compile_model(
    model: Model,
    lr: float,
    focal_gamma: float,
    focal_alpha: float,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=focal_loss(gamma=focal_gamma, alpha=focal_alpha),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )


# ---------------------------------------------------------------------------
# Two-Phase Trainer
# ---------------------------------------------------------------------------

def train_model(
    model: Model,
    backbone: EfficientNetB3,
    X_tab_train: np.ndarray,
    X_img_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_val: np.ndarray,
    X_img_val: np.ndarray,
    y_val: np.ndarray,
    phase1_epochs: int = 15,
    phase2_epochs: int = 15,
    batch_size: int = 32,
    run_phase2: bool = True,
    checkpoint_dir: str = "checkpoints",
    oversample_ratio: float = 0.25,
) -> Tuple[tf.keras.callbacks.History, Optional[tf.keras.callbacks.History]]:
    """
    Two-Phase Training Strategy (matches diagram):

    Phase 1 – Head Training
        Backbone: EfficientNetB3 FROZEN 100%
        Loss    : Focal Loss (γ=2.0, α=0.75) – frozen integers
        LR      : 1e-3 (Adam), Batch=32
        Monitor : val_auc (maximize), Patience=5
        Reduce  : ReduceLROnPlateau 3ep factor=0.5 min_lr=1e-6  Max=15

    Phase 2 – Fine-tuning (unfreeze from layer 300+, ~400 layers trainable)
        Backbone: open from layer 300 →
        Loss    : Focal Loss (γ=2.0, α=0.75) – frozen integers
        LR      : 1e-4 (10× lower than Phase 1)
        Monitor : val_auc (maximize), Patience=7
        Reduce  : ReduceLROnPlateau 3ep factor=0.3 min_lr=1e-7  Max=15
    """
    import os
    from data_preprocessing import oversample_malignant
    from sklearn.utils.class_weight import compute_class_weight

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Oversampling (Layer 1: Malignant × strong aug) ────────────────────
    logger.info("Oversampling training set to target_ratio=%.2f …", oversample_ratio)
    X_img_os, X_tab_os, y_os = oversample_malignant(
        X_img_train, X_tab_train, y_train,
        target_ratio=oversample_ratio,
        strong_aug=True,
    )

    # ── Class weights (Layer 2: ×1.2 – lighter than ×1.5 to avoid triple penalty) ──
    n_neg, n_pos = np.sum(y_os == 0), np.sum(y_os == 1)
    cw = {0: 1.0, 1: (n_neg / n_pos) * 1.2}
    logger.info("Class weights: %s", cw)

    inputs_train = {"image_input": X_img_os, "tabular_input": X_tab_os}
    inputs_val   = {"image_input": X_img_val, "tabular_input": X_tab_val}

    # ── PHASE 1 ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 – Head Training | backbone FROZEN | LR=1e-3 | %d epochs", phase1_epochs)
    callbacks_p1 = _build_callbacks(
        checkpoint_path=os.path.join(checkpoint_dir, "best_phase1.keras"),
        patience_early=5,
        patience_reduce=3,
        reduce_factor=0.5,
        min_lr=1e-6,
    )

    history1 = model.fit(
        inputs_train, y_os,
        validation_data=(inputs_val, y_val),
        epochs=phase1_epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=callbacks_p1,
        verbose=1,
    )

    history2 = None
    if run_phase2:
        # ── PHASE 2 ───────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("PHASE 2 – Fine-tuning | unfreeze from layer %d | LR=1e-4", FINE_TUNE_FROM)

        backbone.trainable = True
        for layer in backbone.layers[:FINE_TUNE_FROM]:
            layer.trainable = False

        _compile_model(model, lr=1e-4, focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA)

        callbacks_p2 = _build_callbacks(
            checkpoint_path=os.path.join(checkpoint_dir, "best_phase2.keras"),
            patience_early=7,
            patience_reduce=3,
            reduce_factor=0.3,
            min_lr=1e-7,
        )

        history2 = model.fit(
            inputs_train, y_os,
            validation_data=(inputs_val, y_val),
            epochs=phase2_epochs,
            batch_size=batch_size,
            class_weight=cw,
            callbacks=callbacks_p2,
            verbose=1,
        )

        best_val_auc = max(history2.history.get("val_auc", [0]))
        logger.info("Best Phase 2 Val AUC: %.4f", best_val_auc)

    return history1, history2


def _build_callbacks(
    checkpoint_path: str,
    patience_early: int,
    patience_reduce: int,
    reduce_factor: float,
    min_lr: float,
):
    return [
        EarlyStopping(
            monitor="val_auc", patience=patience_early,
            restore_best_weights=True, mode="max", verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path, monitor="val_auc",
            save_best_only=True, mode="max", verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc", factor=reduce_factor,
            patience=patience_reduce, min_lr=min_lr, verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Threshold tuning (Grid Search 0.05–0.95, step 0.01)
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    metric: str = "f1",
    min_recall: float = 0.60,
) -> Tuple[float, "pd.DataFrame"]:
    """
    Grid search threshold that maximises F1 (or recall) for Malignant class
    subject to recall >= min_recall.
    Default range: [0.30, 0.45] is typical for ISIC pAUC-optimised models.
    """
    import pandas as pd

    thresholds = np.arange(0.05, 0.95, 0.01)
    rows = []
    for thr in thresholds:
        pred = (y_pred_prob >= thr).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        tn = np.sum((y_true == 0) & (pred == 0))
        recall    = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        rows.append({"threshold": thr, "recall": recall, "precision": precision, "f1": f1})

    df = pd.DataFrame(rows)
    filtered = df[df["recall"] >= min_recall]
    if filtered.empty:
        filtered = df

    if metric == "f1":
        best = filtered.loc[filtered["f1"].idxmax()]
    else:
        best = filtered.loc[filtered["recall"].idxmax()]

    logger.info(
        "Optimal threshold=%.2f  recall=%.4f  precision=%.4f  f1=%.4f",
        best["threshold"], best["recall"], best["precision"], best["f1"],
    )
    return float(best["threshold"]), df


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Model,
    X_tab: np.ndarray,
    X_img: np.ndarray,
    y_true: np.ndarray,
    label_encoder=None,
    auto_tune_threshold: bool = True,
    threshold: float = 0.50,
) -> dict:
    """
    Full evaluation suite:
      - Confusion Matrix (heatmap 2×2)
      - ROC Curve + AUC, pAUC (TPR≥80%) – ISIC metric
      - F1, Precision, Recall for Malignant
    Returns dict of all scalar metrics.
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
    )

    inputs = {"image_input": X_img, "tabular_input": X_tab}
    y_prob = model.predict(inputs, verbose=0).flatten()

    if auto_tune_threshold:
        threshold, _ = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    pauc = compute_pauc(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "auc_roc": round(auc, 4),
        "pauc_tpr80": round(pauc, 4),
        "f1_malignant": round(f1, 4),
        "recall_malignant": round(recall, 4),
        "precision_malignant": round(precision, 4),
        "threshold": round(threshold, 2),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    logger.info(
        "Evaluation → AUC=%.4f | pAUC=%.4f | F1=%.4f | Recall=%.4f | threshold=%.2f",
        auc, pauc, f1, recall, threshold,
    )
    return metrics


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def predict_skin_lesion(
    model: Model,
    image_path: str,
    tabular_row: np.ndarray,
    threshold: float = 0.55,
) -> dict:
    """
    Single-sample inference pipeline (predict_skin_lesion() from diagram).
    tabular_row: preprocessed 1-D float32 vector.
    """
    from data_preprocessing import load_image, preprocess_image

    img = load_image(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = preprocess_image(img)

    img_input = np.expand_dims(img, 0)
    tab_input = np.expand_dims(tabular_row, 0)

    prob = float(model.predict({"image_input": img_input, "tabular_input": tab_input}, verbose=0)[0, 0])
    pred_class = int(prob >= threshold)
    label = "Malignant" if pred_class == 1 else "Benign"
    confidence = prob if pred_class == 1 else 1 - prob

    return {
        "diagnosis": label,
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "probability_malignant": round(prob, 4),
        "threshold": threshold,
    }
