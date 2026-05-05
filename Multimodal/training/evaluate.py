"""
evaluate.py — Bước 6: Đánh giá mô hình + Grid-search Threshold
Metrics: AUC, pAUC (TPR≥0.80), F1, Recall, Precision
Đầu vào:
  /data/output/best_model_isic2024.h5
  /data/processed/tabular_processed.pkl
  /data/processed/encoders.pkl
  /data/splits/{val,test}_idx.npy
Đầu ra:
  /data/eval/metrics.json
  /data/eval/best_threshold.txt
  /data/eval/roc_curve.png
  /data/eval/confusion_matrix.png
  /data/eval/baseline_profile.json   ← dùng cho drift monitoring
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    recall_score, precision_score, confusion_matrix,
)
from PIL import Image
from tqdm import tqdm

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/data/processed")
SPLITS_DIR    = os.environ.get("SPLITS_DIR", "/data/splits")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "/data/output")
EVAL_DIR      = os.environ.get("EVAL_DIR", "/data/eval")
IMAGE_SIZE    = int(os.environ.get("IMAGE_SIZE", "224"))

os.makedirs(EVAL_DIR, exist_ok=True)
IMAGE_DIR   = os.path.join(PROCESSED_DIR, "images")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "best_model_isic2024.h5")


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    focal_loss_fn.__name__ = "focal_loss"
    return focal_loss_fn


def load_split(df, idx, feature_cols):
    records = df.iloc[idx].reset_index(drop=True)
    X_img, X_tab, y_list = [], [], []
    for _, row in tqdm(records.iterrows(), total=len(records)):
        img_path = os.path.join(IMAGE_DIR, f"{row['isic_id']}.png")
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert("RGB"),
                       dtype=np.float32) / 255.0
        X_img.append(img)
        X_tab.append(row[feature_cols].values.astype(np.float32))
        y_list.append(int(row["target"]))
    return (np.array(X_img, dtype=np.float32),
            np.array(X_tab, dtype=np.float32),
            np.array(y_list))


def compute_pauc(y_true, y_score, min_tpr=0.80):
    """pAUC tại vùng TPR ≥ min_tpr (metric chính ISIC 2024)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(tpr[mask], fpr[mask]))


def grid_search_threshold(y_true, y_prob):
    """Tìm threshold tối ưu theo F1 score trên tập validation."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return round(best_thr, 4), round(best_f1, 4)


def main():
    print("Đang tải model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"focal_loss_fn": focal_loss()},
    )

    df       = pd.read_pickle(os.path.join(PROCESSED_DIR, "tabular_processed.pkl"))
    encoders = pickle.load(open(os.path.join(PROCESSED_DIR, "encoders.pkl"), "rb"))
    feature_cols = encoders["feature_cols"]

    idx_val  = np.load(os.path.join(SPLITS_DIR, "val_idx.npy"))
    idx_test = np.load(os.path.join(SPLITS_DIR, "test_idx.npy"))

    print("Tải tập validation...")
    X_img_val, X_tab_val, y_val = load_split(df, idx_val, feature_cols)
    print("Tải tập test...")
    X_img_test, X_tab_test, y_test = load_split(df, idx_test, feature_cols)

    # Dự đoán xác suất
    prob_val  = model.predict(
        {"image_input": X_img_val,  "tabular_input": X_tab_val},
        batch_size=32, verbose=1).squeeze()
    prob_test = model.predict(
        {"image_input": X_img_test, "tabular_input": X_tab_test},
        batch_size=32, verbose=1).squeeze()

    # Grid-search threshold trên validation
    best_thr, val_f1 = grid_search_threshold(y_val, prob_val)
    print(f"\nBest threshold (val F1={val_f1:.4f}): {best_thr}")

    # Metrics trên tập test
    y_pred_test = (prob_test >= best_thr).astype(int)
    metrics = {
        "auc":            round(float(roc_auc_score(y_test, prob_test)), 4),
        "pauc":           round(compute_pauc(y_test, prob_test, 0.80), 4),
        "f1":             round(float(f1_score(y_test, y_pred_test, zero_division=0)), 4),
        "recall":         round(float(recall_score(y_test, y_pred_test, zero_division=0)), 4),
        "precision":      round(float(precision_score(y_test, y_pred_test, zero_division=0)), 4),
        "best_threshold": best_thr,
    }
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(EVAL_DIR, "best_threshold.txt"), "w") as f:
        f.write(str(best_thr))

    # ── ROC Curve ────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"ROC (AUC={metrics['auc']:.4f})")
    plt.axhline(y=0.80, color="red", linestyle="--", label="TPR=0.80 (pAUC)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — ISIC 2024 Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # ── Confusion Matrix ─────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── Baseline Profile cho Drift Monitor ──────────────────────────
    tab_val_df = df.iloc[idx_val][feature_cols].reset_index(drop=True)
    baseline = {
        col: {
            "mean":   float(tab_val_df[col].mean()),
            "std":    float(tab_val_df[col].std()),
            "p25":    float(tab_val_df[col].quantile(0.25)),
            "p50":    float(tab_val_df[col].quantile(0.50)),
            "p75":    float(tab_val_df[col].quantile(0.75)),
        }
        for col in feature_cols
    }
    baseline["_prediction_rate"] = float(prob_val.mean())
    with open(os.path.join(EVAL_DIR, "baseline_profile.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nKết quả lưu vào {EVAL_DIR}")
    print("Evaluate hoàn thành!")


if __name__ == "__main__":
    main()
