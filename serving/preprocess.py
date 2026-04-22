"""
scripts/preprocess.py
=====================
Docker entrypoint for the preprocessing stage.
Reads raw ISIC 2024 HDF5 + CSV → extracts images → saves processed arrays.

Environment variables
---------------------
CSV_PATH   : path to train-metadata.csv
HDF5_PATH  : path to train-image.hdf5
IMAGE_DIR  : directory to save extracted JPEG images
OUTPUT_DIR : directory to save processed .npy arrays + preprocessors
N_BENIGN   : max benign images to sample (all malignant always kept)
"""

import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (
    build_balanced_selected_ids,
    extract_images_from_hdf5,
    prepare_multimodal_data,
    stratified_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("preprocess")


def main() -> None:
    csv_path   = os.environ.get("CSV_PATH",   "/data/train-metadata.csv")
    hdf5_path  = os.environ.get("HDF5_PATH",  "/data/train-image.hdf5")
    image_dir  = os.environ.get("IMAGE_DIR",  "/data/images")
    output_dir = os.environ.get("OUTPUT_DIR", "/data/processed")
    n_benign   = int(os.environ.get("N_BENIGN", "4000"))

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Label-aware HDF5 extraction ────────────────────────────────────
    logger.info("Step 1/3: Selecting image IDs (all malignant + %d benign) …", n_benign)
    selected_ids = build_balanced_selected_ids(csv_path, n_benign=n_benign)

    logger.info("Step 1/3: Extracting %d images from HDF5 …", len(selected_ids))
    extract_images_from_hdf5(hdf5_path, image_dir, selected_ids)

    # ── 2. Filter metadata to extracted images ─────────────────────────────
    logger.info("Step 2/3: Loading metadata and filtering to extracted images …")
    df = pd.read_csv(csv_path)
    available = {
        os.path.splitext(f)[0]
        for f in os.listdir(image_dir)
        if f.lower().endswith(".jpg")
    }
    df = df[df["isic_id"].isin(available)].reset_index(drop=True)
    logger.info("Retained %d rows matching extracted images", len(df))

    # ── 3. Multimodal feature preparation ─────────────────────────────────
    logger.info("Step 3/3: Building multimodal feature arrays …")
    X_tab, X_img, y, feature_cols, label_encoders, scaler, imputer = prepare_multimodal_data(
        df, image_dir, is_training=True
    )

    splits = stratified_split(X_tab, X_img, y)

    # ── Save processed arrays ──────────────────────────────────────────────
    for split_name, (st, si, sy) in splits.items():
        np.save(os.path.join(output_dir, f"X_tab_{split_name}.npy"), st)
        np.save(os.path.join(output_dir, f"X_img_{split_name}.npy"), si)
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"),     sy)
        logger.info(
            "Saved %s split: tab=%s  img=%s  y=%s",
            split_name, st.shape, si.shape, sy.shape,
        )

    # ── Save preprocessors ────────────────────────────────────────────────
    preproc = {
        "feature_cols":    feature_cols,
        "label_encoders":  label_encoders,
        "scaler":          scaler,
        "imputer":         imputer,
    }
    preproc_path = os.path.join(output_dir, "preprocessors.pkl")
    with open(preproc_path, "wb") as f:
        pickle.dump(preproc, f)
    logger.info("Saved preprocessors → %s", preproc_path)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
