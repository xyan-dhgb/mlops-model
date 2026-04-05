"""
Tabular Preprocessing for ISIC 2024 metadata (train-metadata.csv).
Handles: column normalisation, missing values, outlier clipping,
         categorical encoding, and StandardScaler fitting/saving.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_COL = "target"
ID_COL = "isic_id"

EXCLUDE_COLS = [
    LABEL_COL, ID_COL, "patient_id", "attribution", "copyright_license",
    "image_type", "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4",
    "iddx_5", "mel_mitotic_index", "mel_thick_mm", "lesion_id",
]

CATEGORICAL_COLS = ["sex", "anatom_site_general"]


# ── Full preprocessing pipeline ──────────────────────────────────────────────

def preprocess_csv_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and normalise ISIC 2024 metadata.

    Steps:
      1. Normalise column names (lowercase, strip, replace special chars).
      2. Fill missing numerics with median; categoricals with mode.
      3. Clip numeric outliers (IQR × 1.5).

    Args:
        df: Raw DataFrame from train-metadata.csv.

    Returns:
        (df_processed, report) where report is a summary dict.
    """
    report: dict = {}
    df_out = df.copy()

    # ── 1. Column names ───────────────────────────────────────────────────
    old_cols = df_out.columns.tolist()
    df_out.columns = (
        df_out.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    report["column_mapping"] = dict(zip(old_cols, df_out.columns.tolist()))
    report["initial_shape"] = df_out.shape

    # ── 2. Missing values ─────────────────────────────────────────────────
    report["missing_before"] = int(df_out.isnull().sum().sum())

    num_cols = df_out.select_dtypes(include="number").columns.tolist()
    cat_cols = df_out.select_dtypes(include="object").columns.tolist()

    if num_cols:
        imputer_num = SimpleImputer(strategy="median")
        df_out[num_cols] = imputer_num.fit_transform(df_out[num_cols])

    for col in cat_cols:
        if df_out[col].isnull().any():
            df_out[col] = df_out[col].fillna(df_out[col].mode()[0])

    report["missing_after"] = int(df_out.isnull().sum().sum())

    # ── 3. Outlier clipping (IQR) ─────────────────────────────────────────
    outlier_report: dict = {}
    feature_cols = [c for c in num_cols if c not in EXCLUDE_COLS]

    for col in feature_cols:
        q1, q3 = df_out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((df_out[col] < lo) | (df_out[col] > hi)).sum())
        if n_outliers:
            outlier_report[col] = {
                "count": n_outliers,
                "percent": round(n_outliers / len(df_out) * 100, 2),
            }
            df_out[col] = df_out[col].clip(lo, hi)

    report["outliers"] = outlier_report
    report["final_shape"] = df_out.shape

    print(f"[preprocess_csv_data] {report['initial_shape']} → {report['final_shape']}, "
          f"missing {report['missing_before']} → {report['missing_after']}")
    return df_out, report


# ── Feature extraction / encoding ────────────────────────────────────────────

def get_tabular_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (excludes metadata IDs / target)."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in num_cols if c not in EXCLUDE_COLS]


def fit_encoders(df: pd.DataFrame, cat_cols: list[str] | None = None) -> dict:
    """
    Fit a LabelEncoder per categorical column.

    Args:
        df:       DataFrame containing the columns.
        cat_cols: Column names to encode (defaults to CATEGORICAL_COLS).

    Returns:
        Dict mapping column name → fitted LabelEncoder.
    """
    if cat_cols is None:
        cat_cols = CATEGORICAL_COLS

    encoders: dict = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            encoders[col] = le
    return encoders


def encode_categorical(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply fitted LabelEncoders to categorical columns in place."""
    df_out = df.copy()
    for col, le in encoders.items():
        if col in df_out.columns:
            known = set(le.classes_)
            df_out[col] = df_out[col].astype(str).map(
                lambda v: v if v in known else le.classes_[0]
            )
            df_out[col] = le.transform(df_out[col])
    return df_out


def scale_features(
    X: np.ndarray,
    scaler: StandardScaler | None = None,
    fit: bool = True
) -> tuple[np.ndarray, StandardScaler]:
    """
    StandardScale feature matrix.

    Args:
        X:       Feature matrix, shape (N, F).
        scaler:  Existing scaler to re-use (None → create new).
        fit:     If True, fit scaler on X; otherwise only transform.

    Returns:
        (X_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled.astype(np.float32), scaler


# ── Persistence ───────────────────────────────────────────────────────────────

def save_preprocessor(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved preprocessor → {path}")


def load_preprocessor(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
