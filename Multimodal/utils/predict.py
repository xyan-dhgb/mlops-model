"""
Multimodal/utils/predict.py
Inference helper: load model + preprocessor and predict on a single sample.
Used by the serving API and standalone scripts.
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from Multimodal.preprocessing.image_preprocessing import load_image, preprocess_image
from Multimodal.preprocessing.tabular_preprocessing import (
    encode_categorical,
    CATEGORICAL_COLS,
)


def load_model_and_preprocessor(
    model_path: str = "Multimodal/final/best_model.h5",
    preprocessor_path: str = "Multimodal/final/preprocessor.pkl",
):
    """
    Load the saved Keras model and preprocessor bundle.

    Returns:
        (model, preprocessor_dict)
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(preprocessor_path).exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    model = tf.keras.models.load_model(model_path)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def predict_skin_lesion(
    model,
    df_or_path,
    image_path: str,
    tabular_columns: list[str],
    encoders: dict | None = None,
    target_size: tuple = (224, 224),
    threshold: float = 0.5,
    row_index: int = 0,
) -> dict | None:
    """
    Predict malignancy for a single skin lesion.

    Args:
        model:           Trained Keras model.
        df_or_path:      DataFrame or path to metadata CSV.
        image_path:      Path to the lesion image.
        tabular_columns: Feature columns (must match training order).
        encoders:        Dict of fitted LabelEncoders for categorical cols.
        target_size:     Image resize target.
        threshold:       Decision boundary (default 0.5).
        row_index:       Row to use from the DataFrame.

    Returns:
        dict with keys: diagnosis, confidence, probabilities (or None on error).
    """
    # Load data
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
    sample = df.iloc[row_index][tabular_columns].copy()

    # Normalise boolean-like strings
    bool_map = {"true": 1, "false": 0}
    for col in sample.index:
        val = sample[col]
        if isinstance(val, str) and val.strip().lower() in bool_map:
            sample[col] = bool_map[val.strip().lower()]

    # Encode categorical features
    if encoders:
        for col, le in encoders.items():
            if col in sample.index and not pd.isna(sample[col]):
                str_val = str(sample[col])
                if str_val in le.classes_:
                    sample[col] = int(le.transform([str_val])[0])
                else:
                    sample[col] = 0  # fallback to first class

    sample = sample.fillna(0)

    try:
        tab_arr = sample.astype("float32").values
    except Exception as exc:
        print(f"[predict_skin_lesion] Type coercion failed: {exc}")
        return None

    tab_input = tf.constant(tab_arr, dtype=tf.float32)[tf.newaxis]

    # Load and preprocess image
    img_arr = load_image(image_path, target_size=target_size)
    if img_arr is None:
        print(f"[predict_skin_lesion] Could not load image: {image_path}")
        return None

    img_arr = preprocess_image(img_arr)
    if img_arr is None:
        print("[predict_skin_lesion] Image preprocessing failed")
        return None

    img_input = tf.constant(img_arr, dtype=tf.float32)[tf.newaxis]

    # Inference
    pred = model.predict(
        {"image_input": img_input, "tabular_input": tab_input}, verbose=0
    )

    if pred.shape[-1] == 1:
        mal_prob = float(pred[0][0])
        ben_prob = 1.0 - mal_prob
    else:
        ben_prob = float(pred[0][0])
        mal_prob = float(pred[0][1])

    is_malignant = mal_prob >= threshold
    diagnosis = "Malignant (1)" if is_malignant else "Benign (0)"

    return {
        "diagnosis":   diagnosis,
        "confidence":  round(max(mal_prob, ben_prob) * 100, 2),
        "probabilities": {
            "Benign (0)":    round(ben_prob * 100, 2),
            "Malignant (1)": round(mal_prob * 100, 2),
        },
        "raw_score": round(mal_prob, 6),
    }
