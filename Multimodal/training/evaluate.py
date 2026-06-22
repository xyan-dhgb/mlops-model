"""
evaluate.py — Bước 6: Đánh giá + Grid-search Threshold

Đọc từ Local/DVC:
  preprocessed/best_model_isic2024.h5
  splits/test/X_tab_test.npy, X_img_test.npy, y_test.npy
  splits/val/  (cho threshold tuning)
# Cũ: s3://kltn-isic-2024-colab/preprocessed/best_model_isic2024.h5
# Cũ: s3://kltn-isic-2024-colab/splits/...

Ghi lên Local/DVC:
  preprocessed/metrics.json
  preprocessed/best_threshold.txt
  preprocessed/roc_curve.png
  preprocessed/confusion_matrix.png
  preprocessed/baseline_profile.json    ← dùng cho drift monitor
# Cũ: s3://kltn-isic-2024-colab/preprocessed/...

Khớp notebook cell 38 (compute_pauc, find_optimal_threshold):
  pAUC: TPR ≥ 0.80, chuẩn hóa về [0,1] / (1 - min_tpr)
  Threshold: grid 0.05→0.95, bước 0.01, min_recall=0.60
"""
import io
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    recall_score, precision_score, confusion_matrix,
)

import pickle
from tensorflow.keras.models import load_model

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
os.makedirs(os.path.join(DATA_DIR, "final"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "preprocessed"), exist_ok=True)


def focal_loss(gamma=2.0, alpha=0.25):
    def fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    fn.__name__ = "focal_loss"
    return fn


def compute_pauc(y_true, y_score, min_tpr=0.80):
    """pAUC chuẩn hóa — metric chính ISIC 2024. Khớp cell 38."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return 0.0
    return float(getattr(np, "trapezoid", np.trapz)(tpr[mask], fpr[mask]) / (1.0 - min_tpr))


def find_optimal_threshold(y_true, y_prob, min_recall=0.60):
    """Grid-search threshold. Khớp notebook cell 38."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        recall    = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if recall >= min_recall and f1 > best_f1:
            best_f1, best_thr = f1, thr
    return round(best_thr, 4), round(best_f1, 4)


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def main():
    print("=" * 60)
    print("BƯỚC 6: Evaluate")
    print(f"  Bucket: {DATA_DIR}/final/")
    print("=" * 60)

    model = load_model(
        os.path.join(DATA_DIR, "final/best_model_isic2024.h5"),
        compile=False,
        custom_objects={"focal_loss": focal_loss()},
    )
    with open(os.path.join(DATA_DIR, "preprocessed/encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    feature_cols = encoders["feature_cols"]

    # Load test và val splits
    print("\nĐọc test split từ disk...")
    X_tab_test = np.load(os.path.join(DATA_DIR, "splits/test/X_tab_test.npy"))
    X_img_test = np.load(os.path.join(DATA_DIR, "splits/test/X_img_test.npy"), mmap_mode="r")
    y_test     = np.load(os.path.join(DATA_DIR, "splits/test/y_test.npy"))

    print("Đọc val split (threshold tuning)...")
    X_tab_val  = np.load(os.path.join(DATA_DIR, "splits/val/X_tab_val.npy"))
    X_img_val  = np.load(os.path.join(DATA_DIR, "splits/val/X_img_val.npy"), mmap_mode="r")
    y_val      = np.load(os.path.join(DATA_DIR, "splits/val/y_val.npy"))

    # Predict
    prob_val  = model.predict(
        {"image_input": X_img_val,  "tabular_input": X_tab_val},
        batch_size=32, verbose=1).squeeze()
    prob_test = model.predict(
        {"image_input": X_img_test, "tabular_input": X_tab_test},
        batch_size=32, verbose=1).squeeze()

    # Threshold từ val
    best_thr, val_f1 = find_optimal_threshold(y_val, prob_val, min_recall=0.60)
    print(f"\nBest threshold: {best_thr} (val_f1={val_f1:.4f})")

    y_pred = (prob_test >= best_thr).astype(int)
    metrics = {
        "auc":            round(float(roc_auc_score(y_test, prob_test)), 4),
        "pauc_norm":      round(compute_pauc(y_test, prob_test, 0.80), 4),
        "f1":             round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "recall":         round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "precision":      round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "best_threshold": best_thr,
    }
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    # Lưu metrics
    with open(os.path.join(DATA_DIR, "final/metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(DATA_DIR, "final/best_threshold.txt"), "w") as f:
        f.write(str(best_thr))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="steelblue",
            label=f"AUC={metrics['auc']:.4f}")
    ax.axhline(0.80, color="red", linestyle="--", label="TPR=0.80 (pAUC)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve — ISIC 2024 Test Set")
    ax.legend(); ax.grid(alpha=0.3)
    with open(os.path.join(DATA_DIR, "final/roc_curve.png"), "wb") as f:
        f.write(fig_to_bytes(fig))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Benign","Malignant"])
    ax.set_yticklabels(["Benign","Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    with open(os.path.join(DATA_DIR, "final/confusion_matrix.png"), "wb") as f:
        f.write(fig_to_bytes(fig))
    plt.close()

    # Baseline profile cho drift monitor
    baseline = {
        col: {
            "mean": float(X_tab_val[:, i].mean()),
            "std":  float(X_tab_val[:, i].std()),
            "p25":  float(np.percentile(X_tab_val[:, i], 25)),
            "p50":  float(np.percentile(X_tab_val[:, i], 50)),
            "p75":  float(np.percentile(X_tab_val[:, i], 75)),
        }
        for i, col in enumerate(feature_cols)
    }
    baseline["_prediction_rate"] = float(prob_val.mean())
    with open(os.path.join(DATA_DIR, "final/baseline_profile.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nTất cả kết quả → {DATA_DIR}/final/")
    print("\nBước 6 hoàn thành!")


if __name__ == "__main__":
    main()
