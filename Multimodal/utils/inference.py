"""
utils/inference.py
==================
Production-ready inference for a single patient.

Functions
---------
load_model_for_inference   : load model + preprocessor artifacts
predict_single             : predict from (image_path, metadata_row)
predict_from_csv           : batch inference over a metadata CSV
visualize_prediction       : render image + probability bar chart
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from preprocessing.image_preprocessing import load_image, preprocess_image
from models.multimodal_model import focal_loss


# ── ARTIFACT LOADING ─────────────────────────────────────────────────────────

def load_model_for_inference(model_path: str, artifacts_dir: str) -> dict:
    """
    Load the saved model and all preprocessing artifacts.

    Parameters
    ----------
    model_path    : path to .h5 checkpoint
    artifacts_dir : directory containing scaler.pkl, cat_encoders.pkl, etc.

    Returns
    -------
    dict with keys: model, scaler, cat_encoders, label_encoder, feature_names
    """
    def _pkl(name):
        with open(os.path.join(artifacts_dir, name), "rb") as f:
            return pickle.load(f)

    print(f"[inference] Loading model from '{model_path}'…")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"focal_loss_fn": focal_loss()},
        compile=False,
    )

    return {
        "model":         model,
        "scaler":        _pkl("scaler.pkl"),
        "cat_encoders":  _pkl("cat_encoders.pkl"),
        "label_encoder": _pkl("label_encoder.pkl"),
        "feature_names": _pkl("feature_names.pkl"),
    }


# ── SINGLE PREDICTION ────────────────────────────────────────────────────────

def predict_single(image_path:    str,
                   metadata_row:  pd.Series | dict,
                   artifacts:     dict,
                   threshold:     float = 0.5,
                   image_size:    tuple = (224, 224)) -> dict | None:
    """
    Predict Benign / Malignant for one patient.

    Parameters
    ----------
    image_path    : path to the JPEG skin-lesion image
    metadata_row  : single row from train-metadata.csv (as Series or dict)
    artifacts     : output of load_model_for_inference()
    threshold     : decision threshold (default 0.5; tune on validation set)
    image_size    : (H, W) — must match training config

    Returns
    -------
    dict with diagnosis, predicted_class, confidence, probabilities
    or None on error.
    """
    model         = artifacts["model"]
    scaler        = artifacts["scaler"]
    cat_encoders  = artifacts["cat_encoders"]
    feature_names = artifacts["feature_names"]

    # ── Tabular features ─────────────────────────────────────────────────────
    row = pd.Series(metadata_row) if isinstance(metadata_row, dict) else metadata_row.copy()

    # Encode categoricals
    for col, le in cat_encoders.items():
        if col in row.index:
            val = str(row[col])
            known = set(le.classes_)
            val = val if val in known else le.classes_[0]
            row[col] = le.transform([val])[0]

    # Boolean strings → int
    bool_map = {"true": 1, "false": 0}
    for col in feature_names:
        if col in row.index and isinstance(row[col], str):
            row[col] = bool_map.get(row[col].lower().strip(), 0)

    # Build feature vector
    try:
        feat_vec = row[feature_names].fillna(0).values.astype(np.float32)
    except KeyError as e:
        print(f"[predict_single] Missing feature column: {e}")
        return None

    feat_vec_scaled = scaler.transform(feat_vec[np.newaxis, :])
    tab_tensor = tf.constant(feat_vec_scaled, dtype=tf.float32)

    # ── Image ─────────────────────────────────────────────────────────────────
    img_array = load_image(image_path, target_size=image_size)
    if img_array is None:
        print(f"[predict_single] Could not load image: {image_path}")
        return None

    img_array = preprocess_image(img_array)
    if img_array is None:
        return None

    img_tensor = tf.constant(img_array[np.newaxis, ...], dtype=tf.float32)

    # ── Inference ─────────────────────────────────────────────────────────────
    pred = model.predict(
        {"image_input": img_tensor, "tabular_input": tab_tensor},
        verbose=0,
    )

    if pred.shape[-1] == 1:
        prob_mal = float(pred[0, 0])
        prob_ben = 1.0 - prob_mal
    else:
        prob_ben = float(pred[0, 0])
        prob_mal = float(pred[0, 1])

    pred_class = 1 if prob_mal >= threshold else 0
    label      = "Malignant (Ác tính)" if pred_class == 1 else "Benign (Lành tính)"
    confidence = prob_mal * 100 if pred_class == 1 else prob_ben * 100

    return {
        "diagnosis":     label,
        "predicted_class": pred_class,
        "confidence":    round(confidence, 2),
        "probabilities": {
            "Benign (0)":    round(prob_ben * 100, 2),
            "Malignant (1)": round(prob_mal * 100, 2),
        },
        "threshold": threshold,
    }


# ── BATCH INFERENCE ───────────────────────────────────────────────────────────

def predict_from_csv(csv_path:     str,
                     image_dir:    str,
                     artifacts:    dict,
                     threshold:    float = 0.5,
                     image_size:   tuple = (224, 224),
                     output_path:  str | None = None) -> pd.DataFrame:
    """
    Run batch inference over a metadata CSV.

    Parameters
    ----------
    csv_path    : path to (test) metadata CSV
    image_dir   : directory containing the JPEG images
    artifacts   : output of load_model_for_inference()
    output_path : if given, save results CSV here

    Returns
    -------
    DataFrame with columns: isic_id, diagnosis, predicted_class,
                             confidence, prob_benign, prob_malignant
    """
    df = pd.read_csv(csv_path)
    rows = []

    available_ids = {
        os.path.splitext(f)[0]
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    df = df[df["isic_id"].isin(available_ids)].reset_index(drop=True)
    print(f"[predict_from_csv] Running inference on {len(df)} samples…")

    for i, (_, row) in enumerate(df.iterrows()):
        isic_id    = str(row["isic_id"])
        image_path = os.path.join(image_dir, isic_id + ".jpg")

        result = predict_single(image_path, row, artifacts,
                                threshold=threshold, image_size=image_size)
        if result is None:
            continue

        rows.append({
            "isic_id":        isic_id,
            "diagnosis":      result["diagnosis"],
            "predicted_class": result["predicted_class"],
            "confidence":     result["confidence"],
            "prob_benign":    result["probabilities"]["Benign (0)"],
            "prob_malignant": result["probabilities"]["Malignant (1)"],
            "true_target":    row.get("target", np.nan),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}")

    results_df = pd.DataFrame(rows)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"[predict_from_csv] Results saved → {output_path}")

    return results_df


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def visualize_prediction(image_path:        str,
                          prediction_result: dict,
                          heatmap:           np.ndarray | None = None,
                          save_path:         str | None = None):
    """
    Display a skin-lesion image alongside the prediction probability chart.

    Parameters
    ----------
    image_path        : path to the original JPEG
    prediction_result : output of predict_single()
    heatmap           : optional (H', W') Grad-CAM heatmap to overlay
    save_path         : if given, save the figure as PNG
    """
    import cv2

    img = np.array(Image.open(image_path).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ISIC 2024 — Skin Lesion Prediction",
                 fontsize=14, fontweight="bold")

    # Left — image (+ optional heatmap overlay)
    display_img = img.copy()
    if heatmap is not None:
        h, w = img.shape[:2]
        hm_resized = cv2.resize(heatmap, (w, h))
        hm_uint8   = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
        hm_color   = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_rgb     = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        display_img = cv2.addWeighted(img, 0.6, hm_rgb, 0.4, 0)

    axes[0].imshow(display_img)
    axes[0].axis("off")
    diagnosis  = prediction_result["diagnosis"]
    confidence = prediction_result["confidence"]
    color      = "red" if "Malignant" in diagnosis else "green"
    axes[0].set_title(f"{diagnosis}\nConfidence: {confidence:.2f}%",
                      fontsize=13, color=color, fontweight="bold")

    # Right — probability bars
    probs   = prediction_result["probabilities"]
    classes = list(probs.keys())
    values  = list(probs.values())
    bar_colors = ["steelblue" if "Benign" in c else "tomato" for c in classes]

    bars = axes[1].bar(classes, values, color=bar_colors, edgecolor="black")
    axes[1].set_title("Prediction Probabilities", fontweight="bold")
    axes[1].set_ylabel("Probability (%)")
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center", va="bottom", fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_prediction] Saved → {save_path}")
    plt.show()
