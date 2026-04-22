"""
data_preprocessing.py
======================
ISIC 2024 – Multimodal Skin Lesion Classifier
Handles image preprocessing (CLAHE + Gaussian), tabular feature engineering,
oversampling strategy (Malignant ×3, target_ratio=0.25) and DataLoader creation.

Architecture reference: EfficientNetB3 Branch + MLP Branch (see diagram).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE = (224, 224)
IMG_CHANNELS = 3
TRAIN_RATIO = 0.64
VAL_RATIO = 0.16
TEST_RATIO = 0.20
OVERSAMPLE_TARGET_RATIO = 0.25  # Malignant/(Malignant+Benign) after oversampling
CLAHE_CLIP_LIMIT = 2.0
GAUSS_KERNEL = (3, 3)

EXCLUDE_COLS = [
    "target", "isic_id", "patient_id", "attribution", "copyright_license",
    "image_type", "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4",
    "iddx_5", "mel_mitotic_index", "mel_thick_mm", "lesion_id",
]
CATEGORICAL_COLS = ["sex", "anatom_site_general"]

# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(image_path: str, target_size: Tuple[int, int] = TARGET_SIZE) -> Optional[np.ndarray]:
    """Load a JPEG/PNG image and return float32 array in [0, 1]."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return None


def preprocess_image(
    img: np.ndarray,
    apply_clahe: bool = True,
    apply_gaussian: bool = True,
    contrast: float = 1.2,
) -> np.ndarray:
    """
    ISIC 2024 image pipeline:
      1. CLAHE on L-channel (LAB colour space)
      2. Gaussian blur (3×3)
      3. Contrast enhancement
    """
    if img is None:
        raise ValueError("img must not be None")

    uint8 = (img * 255).astype(np.uint8)

    if apply_clahe:
        lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if apply_gaussian:
        uint8 = cv2.GaussianBlur(uint8, GAUSS_KERNEL, 0)

    if contrast != 1.0:
        pil = Image.fromarray(uint8)
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
        uint8 = np.array(pil)

    return uint8.astype(np.float32) / 255.0


def augment_image(
    img: np.ndarray,
    rotation_range: float = 15.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    zoom_range: float = 0.1,
    strong: bool = False,
) -> np.ndarray:
    """
    Data augmentation.
    strong=True → used for Malignant oversampling (rotation ±30°, dual-flip,
    saturation jitter, random crop).
    """
    if img is None:
        raise ValueError("img must not be None")

    uint8 = (img * 255).astype(np.uint8)
    h, w = uint8.shape[:2]

    # Rotation
    angle = np.random.uniform(-30 if strong else -rotation_range,
                               30 if strong else rotation_range)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    uint8 = cv2.warpAffine(uint8, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Flip
    if strong:
        if np.random.rand() > 0.5:
            uint8 = cv2.flip(uint8, 1)
        if np.random.rand() > 0.5:
            uint8 = cv2.flip(uint8, 0)
    elif np.random.rand() > 0.5:
        uint8 = cv2.flip(uint8, 1)

    # Brightness
    factor = np.random.uniform(*brightness_range)
    uint8 = np.clip(uint8.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Zoom (random crop)
    if zoom_range > 0 and np.random.rand() > 0.5:
        crop = int(min(h, w) * zoom_range)
        y1 = np.random.randint(0, crop + 1)
        x1 = np.random.randint(0, crop + 1)
        uint8 = uint8[y1: h - crop + y1, x1: w - crop + x1]
        uint8 = cv2.resize(uint8, (w, h))

    # Strong: saturation jitter
    if strong:
        hsv = cv2.cvtColor(uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return uint8.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# HDF5 image extraction
# ---------------------------------------------------------------------------

def extract_images_from_hdf5(
    hdf5_path: str,
    output_folder: str,
    selected_ids: Optional[set] = None,
) -> str:
    """
    Extract images from ISIC 2024 HDF5 archive.
    If selected_ids is provided only those IDs are extracted (label-aware).
    """
    os.makedirs(output_folder, exist_ok=True)
    extracted, errors = 0, 0

    with h5py.File(hdf5_path, "r") as hf:
        keys = list(hf.keys()) if selected_ids is None else [k for k in hf.keys() if k in selected_ids]
        logger.info("Extracting %d images …", len(keys))

        for isic_id in keys:
            try:
                img_bytes = hf[isic_id][()]
                out_path = os.path.join(output_folder, f"{isic_id}.jpg")
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                extracted += 1
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    logger.warning("Error extracting %s: %s", isic_id, exc)

    logger.info("Extracted %d images, %d errors.", extracted, errors)
    return output_folder


def build_balanced_selected_ids(
    csv_path: str,
    n_benign: int = 4000,
    seed: int = 42,
) -> set:
    """
    Label-aware selection: ALL malignant + random sample of benign.
    Avoids the 'first N images → near-all-benign' pitfall.
    """
    df = pd.read_csv(csv_path)
    mal_ids = df[df["target"] == 1]["isic_id"].tolist()
    ben_ids = df[df["target"] == 0]["isic_id"].tolist()
    rng = np.random.default_rng(seed)
    selected_ben = rng.choice(ben_ids, size=min(n_benign, len(ben_ids)), replace=False).tolist()
    selected = set(mal_ids + selected_ben)
    logger.info("Selected %d malignant + %d benign = %d total", len(mal_ids), len(selected_ben), len(selected))
    return selected


# ---------------------------------------------------------------------------
# Tabular preprocessing
# ---------------------------------------------------------------------------

def build_tabular_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
    label_encoders: Optional[dict] = None,
    imputer: Optional[SimpleImputer] = None,
) -> Tuple[np.ndarray, list, dict, StandardScaler, SimpleImputer]:
    """
    Full tabular pipeline matching the MLP Branch (37 features):
      1. IQR clip (±1.5)
      2. LabelEncoder for sex / anatom_site
      3. StandardScaler (mean=0, std=1)
      4. SimpleImputer (median, 2nd pass)
    Returns (X, feature_cols, label_encoders, scaler, imputer).
    """
    df = df.copy()

    # Identify numeric feature columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    # Encode categorical
    if label_encoders is None:
        label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("unknown")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])
        if col not in feature_cols:
            feature_cols.append(col)

    X = df[feature_cols].values.astype(np.float32)

    # IQR clip
    if fit:
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    else:
        lower = upper = None  # clipping on train bounds only during fitting

    # Imputer
    if fit:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
    else:
        X = imputer.transform(X)

    # Scaler
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X.astype(np.float32), feature_cols, label_encoders, scaler, imputer


# ---------------------------------------------------------------------------
# Oversampling
# ---------------------------------------------------------------------------

def oversample_malignant(
    X_img: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    target_ratio: float = OVERSAMPLE_TARGET_RATIO,
    strong_aug: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer 1 oversampling: duplicate Malignant samples with strong augmentation
    until target_ratio = Malignant / Total is reached.
    ~2393 samples total (4000 Benign + 393 Malignant) → target 25% malignant.
    """
    rng = np.random.default_rng(seed)
    mal_idx = np.where(y == 1)[0]
    n_mal = len(mal_idx)
    n_ben = np.sum(y == 0)

    # How many synthetic malignant samples needed?
    # target_ratio = (n_mal + n_new) / (n_total + n_new)
    # Solving for n_new:
    n_total = len(y)
    n_new = max(0, int((target_ratio * n_total - n_mal) / (1 - target_ratio)))

    if n_new == 0:
        return X_img, X_tab, y

    logger.info("Oversampling: adding %d synthetic Malignant samples (strong_aug=%s)", n_new, strong_aug)

    aug_imgs, aug_tabs, aug_ys = [], [], []
    for _ in range(n_new):
        idx = rng.choice(mal_idx)
        aug_img = augment_image(X_img[idx], strong=strong_aug)
        aug_imgs.append(aug_img)
        aug_tabs.append(X_tab[idx])
        aug_ys.append(1)

    X_img_out = np.concatenate([X_img, np.stack(aug_imgs)], axis=0)
    X_tab_out = np.concatenate([X_tab, np.stack(aug_tabs)], axis=0)
    y_out = np.concatenate([y, np.array(aug_ys)], axis=0)

    # Shuffle
    perm = rng.permutation(len(y_out))
    return X_img_out[perm], X_tab_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def prepare_multimodal_data(
    df: pd.DataFrame,
    image_dir: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    is_training: bool = True,
    label_encoders: Optional[dict] = None,
    scaler: Optional[StandardScaler] = None,
    imputer: Optional[SimpleImputer] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, dict, StandardScaler, SimpleImputer]:
    """
    Full multimodal preparation:
      - Loads & preprocesses images (CLAHE + Gaussian + normalise)
      - Engineers tabular features (IQR clip, encode, scale, impute)
    Returns: (X_tabular, X_image, y, feature_cols, label_encoders, scaler, imputer)
    """
    X_tab, feature_cols, label_encoders, scaler, imputer = build_tabular_features(
        df, fit=is_training, scaler=scaler,
        label_encoders=label_encoders, imputer=imputer,
    )

    y = df["target"].values.astype(np.int32)

    images = []
    valid_mask = []
    for _, row in df.iterrows():
        isic_id = str(row["isic_id"]).strip()
        img_path = os.path.join(image_dir, isic_id + ".jpg")
        if not os.path.exists(img_path):
            valid_mask.append(False)
            images.append(None)
            continue
        img = load_image(img_path, target_size)
        img = preprocess_image(img)
        images.append(img)
        valid_mask.append(True)

    valid_mask = np.array(valid_mask)
    X_img = np.stack([img for img, v in zip(images, valid_mask) if v])
    X_tab = X_tab[valid_mask]
    y = y[valid_mask]

    logger.info(
        "Data ready: %d samples | Benign=%d Malignant=%d | tabular_features=%d",
        len(y), int(np.sum(y == 0)), int(np.sum(y == 1)), X_tab.shape[1],
    )
    return X_tab, X_img, y, feature_cols, label_encoders, scaler, imputer


def stratified_split(
    X_tab: np.ndarray,
    X_img: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.16,
    test_size: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Stratified 64/16/20 split matching the diagram dataloader boxes.
    Returns dict with keys: train, val, test – each a (X_tab, X_img, y) tuple.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss.split(X_tab, y))

    val_fraction = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss2.split(X_tab[trainval_idx], y[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    return {
        "train": (X_tab[train_idx], X_img[train_idx], y[train_idx]),
        "val": (X_tab[val_idx], X_img[val_idx], y[val_idx]),
        "test": (X_tab[test_idx], X_img[test_idx], y[test_idx]),
    }
