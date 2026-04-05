"""
Multimodal/utils/serving.py
FastAPI inference server for the ISIC 2024 multimodal model.

Endpoints:
  GET  /health          – liveness check
  POST /predict         – single lesion prediction (JSON metadata + image upload)
  POST /predict/batch   – batch prediction from JSON list
"""

import os
import io
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# ── Lazy-loaded globals (set at startup) ─────────────────────────────────────
_model = None
_preprocessor = None

app = FastAPI(
    title="ISIC 2024 Skin Lesion Classifier",
    description="Multimodal CNN+MLP model — Binary: Benign (0) vs Malignant (1)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_artifacts():
    global _model, _preprocessor

    model_path = os.getenv("MODEL_PATH", "Multimodal/final/best_model.h5")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "Multimodal/final/preprocessor.pkl")

    if not Path(model_path).exists():
        raise RuntimeError(f"Model not found: {model_path}")
    if not Path(preprocessor_path).exists():
        raise RuntimeError(f"Preprocessor not found: {preprocessor_path}")

    _model = tf.keras.models.load_model(model_path)
    with open(preprocessor_path, "rb") as f:
        _preprocessor = pickle.load(f)

    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Preprocessor loaded from {preprocessor_path}")
    print(f"   Tabular features ({len(_preprocessor['tabular_cols'])}): "
          f"{_preprocessor['tabular_cols'][:5]} ...")


# ── Schemas ───────────────────────────────────────────────────────────────────

class TabularFeatures(BaseModel):
    """Flat dict of numeric metadata features matching training columns."""
    features: dict = Field(..., description="Feature name → value mapping")


class PredictionResponse(BaseModel):
    isic_id: Optional[str] = None
    diagnosis: str
    malignant_probability: float
    benign_probability: float
    confidence_pct: float
    threshold: float = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # CLAHE via OpenCV
    try:
        import cv2
        img_u8 = (arr * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    except Exception:
        pass  # Return raw normalised image if OpenCV unavailable

    return arr


def _build_tabular_input(feature_dict: dict) -> np.ndarray:
    cols = _preprocessor["tabular_cols"]
    encoders = _preprocessor.get("encoders", {})

    row = []
    for col in cols:
        val = feature_dict.get(col, 0.0)
        if col in encoders:
            le = encoders[col]
            str_val = str(val)
            if str_val in le.classes_:
                val = int(le.transform([str_val])[0])
            else:
                val = 0
        try:
            row.append(float(val))
        except (TypeError, ValueError):
            row.append(0.0)

    return np.array(row, dtype=np.float32)[np.newaxis]


def _run_inference(tab_input: np.ndarray, img_input: np.ndarray, threshold: float = 0.5) -> dict:
    pred = _model.predict(
        {"image_input": img_input, "tabular_input": tab_input}, verbose=0
    )

    if pred.shape[-1] == 1:
        mal_prob = float(pred[0][0])
        ben_prob = 1.0 - mal_prob
    else:
        ben_prob = float(pred[0][0])
        mal_prob = float(pred[0][1])

    is_malignant = mal_prob >= threshold
    return {
        "diagnosis": "Malignant" if is_malignant else "Benign",
        "malignant_probability": round(mal_prob, 6),
        "benign_probability": round(ben_prob, 6),
        "confidence_pct": round(max(mal_prob, ben_prob) * 100, 2),
        "threshold": threshold,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "n_features": len(_preprocessor["tabular_cols"]) if _preprocessor else 0,
    }


@app.get("/features", tags=["System"])
def list_features():
    """Return expected tabular feature names."""
    if _preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    return {"tabular_columns": _preprocessor["tabular_cols"]}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    isic_id: Optional[str] = None,
    threshold: float = 0.5,
    features: str = "{}",       # JSON string of metadata features
    image: UploadFile = File(...),
):
    """
    Predict malignancy from skin lesion image + metadata features.

    - **image**: JPEG/PNG skin lesion image
    - **features**: JSON string mapping feature names to values
    - **threshold**: decision threshold (default 0.5)
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Parse features
    import json
    try:
        feature_dict = json.loads(features)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid features JSON: {exc}")

    # Process image
    img_bytes = await image.read()
    try:
        img_arr = _preprocess_image(img_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {exc}")

    img_input = img_arr[np.newaxis]                    # (1, H, W, 3)
    tab_input = _build_tabular_input(feature_dict)     # (1, F)

    result = _run_inference(tab_input, img_input, threshold)
    result["isic_id"] = isic_id
    return result


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(
    threshold: float = 0.5,
    images: list[UploadFile] = File(...),
    features_list: str = "[]",  # JSON array of feature dicts
):
    """
    Batch prediction: N images + N metadata dicts (same order).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import json
    try:
        feat_list = json.loads(features_list)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid features_list JSON: {exc}")

    if len(images) != len(feat_list):
        raise HTTPException(
            status_code=422,
            detail=f"Mismatch: {len(images)} images vs {len(feat_list)} feature dicts"
        )

    results = []
    for img_file, feat_dict in zip(images, feat_list):
        img_bytes = await img_file.read()
        try:
            img_arr = _preprocess_image(img_bytes)[np.newaxis]
        except Exception as exc:
            results.append({"error": str(exc)})
            continue

        tab_arr = _build_tabular_input(feat_dict)
        results.append(_run_inference(tab_arr, img_arr, threshold))

    return {"predictions": results, "count": len(results)}
