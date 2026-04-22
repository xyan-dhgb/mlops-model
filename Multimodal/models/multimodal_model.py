"""
models/multimodal_model.py
===========================
EfficientNetB3 + Tabular MLP multimodal model for ISIC 2024 binary
skin-lesion classification (Benign=0 / Malignant=1).

Public API
----------
focal_loss(gamma, alpha)                        → Keras loss function
build_multimodal_model(tabular_shape, ...)      → (model, backbone)
unfreeze_and_recompile(model, backbone, ...)    → model  [Phase 2 fine-tune]
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate,
    GlobalAveragePooling2D, BatchNormalization,
)

# Name of the last Conv layer in EfficientNetB3 (used by Grad-CAM)
EFFICIENTNET_LAST_CONV = "top_activation"


# ── FOCAL LOSS ───────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Binary Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : focusing parameter  (2.0 is standard)
    alpha : weight for positive class (Malignant)
            Typically > 0.5 when Malignant is the minority class.
    """
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = -y_true * tf.math.log(y_pred) \
              - (1 - y_true) * tf.math.log(1 - y_pred)

        p_t           = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight  = tf.pow(1.0 - p_t, gamma)
        alpha_t       = y_true * alpha + (1 - y_true) * (1 - alpha)

        return tf.reduce_mean(alpha_t * focal_weight * bce)

    return focal_loss_fn


# ── MODEL BUILDER ────────────────────────────────────────────────────────────

def build_multimodal_model(
    tabular_shape: tuple,
    image_shape:   tuple  = (224, 224, 3),
    num_classes:   int    = 2,
    freeze_backbone:  bool  = True,
    fine_tune_from:   int   = 300,
    focal_gamma:      float = 2.0,
    focal_alpha:      float = 0.75,
) -> tuple:
    """
    Build the EfficientNetB3-based multimodal model.

    Architecture
    ────────────
    Image branch  : EfficientNetB3 (pretrained ImageNet)
                    → GlobalAveragePooling2D
                    → BN → Dense(256) → Dropout(0.4)
                    → Dense(128) → Dropout(0.3)

    Tabular branch: Dense(128) → BN → Dropout(0.3)
                    → Dense(64) → Dropout(0.2)
                    → Dense(32)

    Fusion        : Concatenate
                    → Dense(256) → BN → Dropout(0.4)
                    → Dense(128) → Dropout(0.3)
                    → Dense(64)  → Dropout(0.2)
                    → Dense(1, sigmoid)   [binary output]

    Loss          : Focal Loss (gamma=2, alpha=0.75)
    Metrics       : AUC, pAUC, Recall, Precision  — NOT accuracy

    Parameters
    ----------
    tabular_shape   : (n_features,)  — number of tabular input features
    image_shape     : (H, W, C)      — default (224, 224, 3)
    freeze_backbone : True  → Phase 1 (head only)
                      False → Phase 2 (fine-tune)
    fine_tune_from  : layer index from which backbone is trainable (Phase 2)
    focal_gamma     : Focal Loss gamma
    focal_alpha     : Focal Loss alpha (weight for Malignant class)

    Returns
    -------
    (model, backbone)   — backbone is exposed for unfreeze_and_recompile()
    """
    # ── IMAGE BRANCH ─────────────────────────────────────────────────────────
    image_input = Input(shape=image_shape, name="image_input")
    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        pooling=None,
    )

    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False

    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    # ── TABULAR BRANCH ───────────────────────────────────────────────────────
    tabular_input = Input(shape=tabular_shape, name="tabular_input")
    t = Dense(128, activation="relu")(tabular_input)
    t = BatchNormalization()(t)
    t = Dropout(0.3)(t)
    t = Dense(64, activation="relu")(t)
    t = Dropout(0.2)(t)
    t = Dense(32, activation="relu")(t)

    # ── FUSION ───────────────────────────────────────────────────────────────
    combined = Concatenate()([x, t])
    z = Dense(256, activation="relu")(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = Dense(128, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.2)(z)

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    output = Dense(1, activation="sigmoid", name="output")(z)
    model  = Model(inputs=[image_input, tabular_input], outputs=output)

    # ── COMPILE ──────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=focal_gamma, alpha=focal_alpha),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(
                name="pauc",
                num_thresholds=1000,
                summation_method="interpolation",
                curve="ROC",
            ),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            # NOTE: accuracy deliberately omitted — misleading on 97/3 imbalance
        ],
    )

    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    total_params = sum(
        tf.size(w).numpy() for w in model.weights
    )
    print(f"[build_multimodal_model] "
          f"Total params: {total_params:,}  |  "
          f"Trainable: {trainable_params:,}  |  "
          f"Backbone frozen: {freeze_backbone}")

    return model, backbone


# ── PHASE-2 FINE-TUNING ──────────────────────────────────────────────────────

def unfreeze_and_recompile(
    model,
    backbone,
    fine_tune_lr:   float = 1e-4,
    fine_tune_from: int   = 300,
    focal_gamma:    float = 2.0,
    focal_alpha:    float = 0.75,
):
    """
    Unfreeze the top layers of EfficientNetB3 for Phase 2 fine-tuning.

    Parameters
    ----------
    model           : compiled Keras model from build_multimodal_model()
    backbone        : EfficientNetB3 sub-model (returned alongside model)
    fine_tune_lr    : lower LR for fine-tuning (default 1e-4)
    fine_tune_from  : freeze layers[:fine_tune_from], unfreeze the rest
    focal_gamma/alpha : keep same Focal Loss settings

    Returns
    -------
    Recompiled model (same object, mutated in place, but returned for clarity).
    """
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_from]:
        layer.trainable = False

    n_total    = len(backbone.layers)
    n_unfreeze = sum(1 for l in backbone.layers if l.trainable)
    print(f"[unfreeze] EfficientNetB3: {n_total} layers total | "
          f"{n_unfreeze} unfrozen (from layer {fine_tune_from})")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=focal_loss(gamma=focal_gamma, alpha=focal_alpha),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model
