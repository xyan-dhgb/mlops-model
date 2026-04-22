"""
utils/metrics.py
=================
Evaluation utilities for the ISIC 2024 multimodal model.

Functions
---------
compute_pauc            : partial AUC at min TPR = 0.80
find_optimal_threshold  : grid-search threshold maximising F1 (or precision)
evaluate_model          : full evaluation report → dict
plot_training_history   : loss / AUC / recall / precision curves
plot_roc_pr             : ROC + Precision-Recall curves
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    f1_score,
)


# ── PAUC ─────────────────────────────────────────────────────────────────────

def compute_pauc(y_true: np.ndarray,
                 y_pred_prob: np.ndarray,
                 min_tpr: float = 0.80) -> float:
    """
    Partial AUC restricted to TPR ≥ min_tpr.

    This is the primary competition metric for ISIC 2024:
    maximise sensitivity while keeping specificity reasonable.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return 0.0
    return float(auc(fpr[mask], tpr[mask]))


# ── THRESHOLD TUNING ─────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray,
                            y_pred_prob: np.ndarray,
                            metric: str = "f1",
                            min_recall: float = 0.60) -> tuple[float, float]:
    """
    Grid-search the decision threshold.

    Parameters
    ----------
    metric     : 'f1' or 'precision'
    min_recall : only consider thresholds where Recall ≥ min_recall

    Returns
    -------
    (best_threshold, best_score)
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thr, best_score = 0.5, 0.0

    for thr in thresholds:
        y_pred = (y_pred_prob >= thr).astype(int)

        recall = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
        if recall < min_recall:
            continue

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thr   = thr

    print(f"[find_optimal_threshold] Best threshold={best_thr:.2f}  "
          f"{metric}={best_score:.4f}")
    return float(best_thr), float(best_score)


# ── FULL EVALUATION ──────────────────────────────────────────────────────────

def evaluate_model(model,
                   test: dict,
                   threshold: float = 0.5,
                   min_tpr_pauc: float = 0.80,
                   save_path: str | None = None) -> dict:
    """
    Run full evaluation on the test set.

    Parameters
    ----------
    model      : fitted Keras model
    test       : dict with 'images', 'tabular', 'labels'
    threshold  : decision threshold (from find_optimal_threshold)
    save_path  : if given, save results JSON here

    Returns
    -------
    dict with auc, pauc, f1_malignant, recall, precision, confusion_matrix
    """
    X_img = test["images"]
    X_tab = test["tabular"]
    y_true = test["labels"]

    y_pred_prob = model.predict(
        {"image_input": X_img, "tabular_input": X_tab},
        verbose=0,
    ).flatten()

    y_pred = (y_pred_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_prob))
    pauc    = compute_pauc(y_true, y_pred_prob, min_tpr=min_tpr_pauc)
    f1_mal  = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
    )

    results = {
        "threshold":       threshold,
        "roc_auc":         roc_auc,
        f"pauc_tpr{min_tpr_pauc}": pauc,
        "f1_malignant":    f1_mal,
        "recall_malignant":  report["Malignant"]["recall"],
        "precision_malignant": report["Malignant"]["precision"],
        "confusion_matrix":  cm,
        "classification_report": report,
    }

    print(f"[evaluate_model]  AUC={roc_auc:.4f}  "
          f"pAUC={pauc:.4f}  "
          f"F1(Malignant)={f1_mal:.4f}  "
          f"Threshold={threshold:.2f}")

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[evaluate_model] Results saved → {save_path}")

    return results


# ── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_training_history(history1, history2=None, save_path: str | None = None):
    """
    Plot AUC, Recall, Precision, and Loss curves.
    Accuracy is intentionally excluded (misleading on 97/3 imbalanced data).
    """
    def _get(h, key):
        return h.history.get(key, [])

    phases = [(history1, "Phase 1")]
    if history2 is not None:
        phases.append((history2, "Phase 2"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History — EfficientNetB3 Multimodal ISIC 2024\n"
                 "(Accuracy omitted — meaningless on 97/3 imbalance)",
                 fontsize=13, fontweight="bold")

    keys_titles = [
        ("auc",       "val_auc",       "AUC"),
        ("recall",    "val_recall",    "Recall"),
        ("precision", "val_precision", "Precision"),
        ("loss",      "val_loss",      "Loss"),
    ]

    for ax, (tr_key, val_key, title) in zip(axes.flatten(), keys_titles):
        offset = 0
        for hist, label in phases:
            tr_vals  = _get(hist, tr_key)
            val_vals = _get(hist, val_key)
            epochs   = range(offset, offset + len(tr_vals))
            if tr_vals:
                ax.plot(epochs, tr_vals,  label=f"{label} Train")
            if val_vals:
                ax.plot(epochs, val_vals, label=f"{label} Val", linestyle="--")
            offset += len(tr_vals)

        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_training_history] Saved → {save_path}")
    plt.show()


def plot_roc_pr(y_true: np.ndarray,
                y_pred_prob: np.ndarray,
                threshold: float = 0.5,
                save_path: str | None = None):
    """Plot ROC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc      = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc        = auc(rec, prec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Evaluation — ISIC 2024", fontsize=14, fontweight="bold")

    # ROC
    ax1.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.axvline(x=fpr[np.argmin(np.abs(tpr - 0.80))], color="red",
                linestyle=":", label="TPR = 0.80 (pAUC boundary)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()

    # Precision-Recall
    ax2.plot(rec, prec, lw=2, color="darkorange",
             label=f"PR AUC = {pr_auc:.4f}")
    ax2.axhline(y=np.mean(y_true), color="navy", linestyle="--",
                label=f"Baseline = {np.mean(y_true):.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve\n(AUC-PR high = good with imbalanced data)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_roc_pr] Saved → {save_path}")
    plt.show()
