"""
training/train.py
==================
Two-phase training loop for the ISIC 2024 multimodal model.

Public API
----------
train_model(cfg, train, val, model, backbone) → (model, history1, history2)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from models.multimodal_model import unfreeze_and_recompile
from preprocessing.image_preprocessing import oversample_malignant
from utils.metrics import compute_pauc, find_optimal_threshold, evaluate_model


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _class_weights(y: np.ndarray, multiplier: float = 1.2) -> dict:
    """
    Compute balanced class weights with an optional penalty multiplier
    for the positive (Malignant) class.

    Using ×1.2 instead of ×1.5 avoids triple-penalty bias when combined
    with Focal Loss and oversampling.
    """
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    total = n_neg + n_pos

    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos) * multiplier
    print(f"[class_weights] Benign={w_neg:.3f}  Malignant={w_pos:.3f}")
    return {0: w_neg, 1: w_pos}


def _make_callbacks(checkpoint_path: str,
                    monitor: str = "val_auc",
                    patience: int = 8) -> list:
    """Build standard callback list for one training phase."""
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(2, patience // 3),
            mode="max",
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ── MAIN TRAINING FUNCTION ────────────────────────────────────────────────────

def train_model(
    cfg:      dict,
    train:    dict,
    val:      dict,
    model,
    backbone,
) -> tuple:
    """
    Two-phase training for the multimodal ISIC 2024 model.

    Phase 1 — EfficientNetB3 backbone FROZEN
        Train the fusion head + tabular branch only.
        Fast convergence, prevents catastrophic forgetting.

    Phase 2 — Backbone UNFROZEN from layer 300
        Fine-tune the top layers of EfficientNetB3.
        Lower LR (1e-4) to avoid destroying ImageNet features.

    Parameters
    ----------
    cfg      : full config dict (from train_config.yaml)
    train    : dict with 'images', 'tabular', 'labels'
    val      : same
    model    : Keras model (Phase 1 compiled)
    backbone : EfficientNetB3 sub-model

    Returns
    -------
    (model, history1, history2)
    """
    p1_cfg   = cfg["training"]["phase1"]
    p2_cfg   = cfg["training"]["phase2"]
    imb_cfg  = cfg["imbalance"]
    loss_cfg = cfg["loss"]

    X_img_train = train["images"]
    X_tab_train = train["tabular"]
    y_train     = train["labels"]

    X_img_val   = val["images"]
    X_tab_val   = val["tabular"]
    y_val       = val["labels"]

    # ── Oversampling ─────────────────────────────────────────────────────────
    X_img_os, X_tab_os, y_os = oversample_malignant(
        X_img_train, X_tab_train, y_train,
        target_ratio=imb_cfg.get("oversample_ratio", 0.25),
        strong_aug=imb_cfg.get("strong_aug_for_minority", True),
    )
    print(f"[train] After oversampling: {len(y_os)} samples  "
          f"(Benign={np.sum(y_os==0)}, Malignant={np.sum(y_os==1)})")

    class_weights = _class_weights(
        y_os,
        multiplier=imb_cfg.get("class_weight_multiplier", 1.2),
    )

    # ── PHASE 1 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — Training head  (backbone FROZEN)")
    print(f"  Epochs: {p1_cfg['epochs']}  |  "
          f"Batch: {p1_cfg['batch_size']}  |  "
          f"LR: {p1_cfg['learning_rate']}")
    print("=" * 60)

    callbacks1 = _make_callbacks(
        checkpoint_path=p1_cfg["checkpoint_path"],
        monitor=p1_cfg.get("monitor", "val_auc"),
        patience=p1_cfg.get("patience", 8),
    )

    history1 = model.fit(
        {"image_input": X_img_os, "tabular_input": X_tab_os},
        y_os,
        validation_data=(
            {"image_input": X_img_val, "tabular_input": X_tab_val},
            y_val,
        ),
        epochs=p1_cfg["epochs"],
        batch_size=p1_cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks1,
        verbose=1,
    )

    # ── PHASE 2 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2 — Fine-tuning backbone from layer {p2_cfg.get('fine_tune_from', 300)}")
    print(f"  Epochs: {p2_cfg['epochs']}  |  "
          f"Batch: {p2_cfg['batch_size']}  |  "
          f"LR: {p2_cfg['learning_rate']}")
    print("=" * 60)

    model = unfreeze_and_recompile(
        model,
        backbone,
        fine_tune_lr=p2_cfg["learning_rate"],
        fine_tune_from=p2_cfg.get("fine_tune_from", 300),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.75),
    )

    callbacks2 = _make_callbacks(
        checkpoint_path=p2_cfg["checkpoint_path"],
        monitor=p2_cfg.get("monitor", "val_auc"),
        patience=p2_cfg.get("patience", 5),
    )

    history2 = model.fit(
        {"image_input": X_img_os, "tabular_input": X_tab_os},
        y_os,
        validation_data=(
            {"image_input": X_img_val, "tabular_input": X_tab_val},
            y_val,
        ),
        epochs=p2_cfg["epochs"],
        batch_size=p2_cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks2,
        verbose=1,
    )

    print("\n[train] Training complete.")
    return model, history1, history2
