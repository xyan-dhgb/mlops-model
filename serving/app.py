"""
serving/app.py
==============
FastAPI REST inference server for ISIC 2024 Multimodal Classifier.

Endpoints
─────────
GET  /health          – liveness probe
GET  /model/info      – model metadata + threshold
POST /predict         – single-sample skin lesion prediction
POST /predict/batch   – batch prediction (up to 32 samples)
GET  /metrics         – Prometheus metrics

POST /predict  request (multipart/form-data)
  image     : JPEG/PNG file
  metadata  : JSON string with tabular fields

POST /predict  response
  {
    "diagnosis":              "Benign" | "Malignant",
    "predicted_class":        0 | 1,
    "probability_malignant":  float,
    "confidence":             float,
    "threshold":              float,
    "gradcam_b64":            base64 PNG overlay
  }
"""

from __future__ import annotations

import base64
import io
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logger = logging.getLogger("serving")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
_model: Optional[tf.keras.Model] = None
_preproc: Optional[dict] = None
_threshold: float = 0.55
_EFFICIENTNET_LAST_CONV = "top_activation"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + preprocessors at startup."""
    global _model, _preproc, _threshold

    model_path  = os.environ.get("MODEL_PATH",          "/models/multimodal_model.keras")
    preproc_path = os.environ.get("PREPROCESSORS_PATH", "/models/preprocessors.pkl")
    _threshold  = float(os.environ.get("THRESHOLD",     "0.55"))

    logger.info("Loading model from %s …", model_path)
    _model = tf.keras.models.load_model(
        model_path,
        custom_objects={"focal_loss_fn": _dummy_loss},
        compile=False,
    )
    logger.info("Model loaded. Output shape: %s", _model.output_shape)

    logger.info("Loading preprocessors from %s …", preproc_path)
    with open(preproc_path, "rb") as f:
        _preproc = pickle.load(f)

    if "threshold" in _preproc:
        _threshold = _preproc["threshold"]
    logger.info("Preprocessors loaded. threshold=%.2f", _threshold)

    yield  # app runs

    logger.info("Shutting down serving app.")


def _dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


app = FastAPI(
    title="ISIC 2024 Skin Lesion Classifier",
    description="Multimodal EfficientNetB3 + MLP – Benign / Malignant prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    diagnosis:              str
    predicted_class:        int
    probability_malignant:  float
    confidence:             float
    threshold:              float
    gradcam_b64:            Optional[str] = None
    inference_ms:           float


class ModelInfo(BaseModel):
    model_path:    str
    threshold:     float
    n_features:    int
    feature_cols:  List[str]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image_bytes(data: bytes, target_size=(224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # CLAHE
    uint8 = (arr * 255).astype(np.uint8)
    lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    uint8 = cv2.GaussianBlur(uint8, (3, 3), 0)
    return uint8.astype(np.float32) / 255.0


def _tabular_from_dict(meta: dict) -> np.ndarray:
    """Convert JSON metadata dict → preprocessed float32 vector."""
    import pandas as pd
    preproc = _preproc
    feature_cols   = preproc["feature_cols"]
    label_encoders = preproc["label_encoders"]
    scaler         = preproc["scaler"]
    imputer        = preproc["imputer"]

    row = {col: meta.get(col, np.nan) for col in feature_cols}
    df  = pd.DataFrame([row])

    for col, le in label_encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            val = val if val in set(le.classes_) else le.classes_[0]
            df[col] = le.transform([val])

    X = df.values.astype(np.float32)
    X = imputer.transform(X)
    X = scaler.transform(X)
    return X.astype(np.float32)


def _grad_cam_overlay(img_arr: np.ndarray, tab_arr: np.ndarray) -> str:
    """Compute Grad-CAM and return base64-encoded PNG overlay."""
    try:
        last_conv = _model.get_layer(_EFFICIENTNET_LAST_CONV)
        grad_model = tf.keras.models.Model(
            inputs=_model.inputs,
            outputs=[last_conv.output, _model.output],
        )
        img_t = tf.constant(img_arr[np.newaxis], dtype=tf.float32)
        tab_t = tf.constant(tab_arr, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model([img_t, tab_t])
            loss = preds[:, 0]

        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

        h, w    = img_arr.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            (img_arr * 255).astype(np.uint8), 0.6, heatmap, 0.4, 0
        )
        pil_out = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        logger.warning("Grad-CAM failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=ModelInfo, tags=["ops"])
async def model_info():
    if _model is None or _preproc is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return ModelInfo(
        model_path=os.environ.get("MODEL_PATH", "unknown"),
        threshold=_threshold,
        n_features=len(_preproc["feature_cols"]),
        feature_cols=_preproc["feature_cols"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    image:    UploadFile = File(..., description="Skin lesion image (JPEG/PNG)"),
    metadata: str        = Form("{}", description="JSON string of tabular metadata fields"),
):
    """
    Multimodal prediction endpoint.
    Accepts image file + JSON metadata string, returns P(Malignant).
    """
    if _model is None or _preproc is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")

    import json
    t0 = time.perf_counter()

    # Parse inputs
    img_bytes = await image.read()
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Invalid JSON metadata: {exc}")

    # Preprocess
    img_arr = _load_image_bytes(img_bytes)
    tab_arr = _tabular_from_dict(meta_dict)

    # Inference
    img_t = img_arr[np.newaxis].astype(np.float32)
    tab_t = tab_arr.astype(np.float32)
    prob  = float(_model.predict({"image_input": img_t, "tabular_input": tab_t}, verbose=0)[0, 0])

    pred_class = int(prob >= _threshold)
    label      = "Malignant" if pred_class == 1 else "Benign"
    confidence = prob if pred_class == 1 else 1.0 - prob

    # Grad-CAM overlay
    gradcam_b64 = _grad_cam_overlay(img_arr, tab_t)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Predict: %s  prob=%.4f  threshold=%.2f  ms=%.1f",
        label, prob, _threshold, elapsed_ms,
    )

    return PredictResponse(
        diagnosis=label,
        predicted_class=pred_class,
        probability_malignant=round(prob, 4),
        confidence=round(confidence, 4),
        threshold=_threshold,
        gradcam_b64=gradcam_b64,
        inference_ms=round(elapsed_ms, 1),
    )


@app.post("/predict/batch", tags=["inference"])
async def predict_batch(images: List[UploadFile] = File(...)):
    """Batch endpoint – images only, metadata assumed zeros for speed."""
    if _model is None or _preproc is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    if len(images) > 32:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Max batch size is 32")

    results = []
    n_feat  = len(_preproc["feature_cols"])
    dummy_tab = np.zeros((1, n_feat), dtype=np.float32)

    for img_file in images:
        data    = await img_file.read()
        img_arr = _load_image_bytes(data)
        prob    = float(_model.predict(
            {"image_input": img_arr[np.newaxis], "tabular_input": dummy_tab}, verbose=0
        )[0, 0])
        pred_class = int(prob >= _threshold)
        results.append({
            "filename": img_file.filename,
            "diagnosis": "Malignant" if pred_class == 1 else "Benign",
            "predicted_class": pred_class,
            "probability_malignant": round(prob, 4),
        })

    return JSONResponse(content={"predictions": results})
