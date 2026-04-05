"""
Multimodal/utils/metrics.py
Custom metrics and evaluation helpers for ISIC 2024.
pAUC @ 80% TPR is the competition's primary metric.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt


def partial_auc(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = 0.80) -> float:
    """
    Compute Partial AUC normalised to [0, 1] over TPR ≥ min_tpr.
    This is the primary metric for the ISIC 2024 challenge.

    Args:
        y_true:  Binary labels (0/1).
        y_score: Predicted probabilities for class 1 (malignant).
        min_tpr: TPR lower bound (default 0.80).

    Returns:
        Normalised pAUC in [0, 1], or nan if too few points above threshold.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return float("nan")
    return float(auc(fpr[mask], tpr[mask]) / (1.0 - min_tpr))


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute the full metrics suite used in training reports.

    Returns:
        dict with keys: accuracy, auc, pauc_80, sensitivity, specificity, ppv, npv
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-9)    # Recall / TPR
    specificity = tn / (tn + fp + 1e-9)    # TNR
    ppv = tp / (tp + fp + 1e-9)            # Precision
    npv = tn / (tn + fn + 1e-9)

    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "auc":         float(roc_auc_score(y_true, y_score)),
        "pauc_80":     partial_auc(y_true, y_score),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv":         float(ppv),
        "npv":         float(npv),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, title: str = "ROC Curve") -> None:
    """Plot ROC curve with pAUC@80% region highlighted."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    pauc = partial_auc(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {auc_score:.4f})")

    # Shade pAUC region
    mask = tpr >= 0.80
    ax.fill_between(fpr[mask], tpr[mask], alpha=0.2, color="orange",
                    label=f"pAUC@80% = {pauc:.4f}")

    ax.axhline(0.80, color="orange", linestyle="--", alpha=0.6, label="TPR = 80%")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_metrics_table(metrics: dict) -> None:
    """Pretty-print the metrics dict."""
    print("\n" + "=" * 45)
    print(f"  {'Metric':<20}  {'Value':>10}")
    print("=" * 45)
    order = ["accuracy", "auc", "pauc_80", "sensitivity", "specificity", "ppv", "npv"]
    for key in order:
        if key in metrics:
            print(f"  {key:<20}  {metrics[key]:>10.4f}")
    print("-" * 45)
    print(f"  TP={metrics.get('tp',0):4d}  FP={metrics.get('fp',0):4d}  "
          f"TN={metrics.get('tn',0):4d}  FN={metrics.get('fn',0):4d}")
    print("=" * 45)
