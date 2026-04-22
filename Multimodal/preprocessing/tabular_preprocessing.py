"""
preprocessing/tabular_preprocessing.py
=======================================
Tabular (metadata) preprocessing for ISIC 2024.

Functions
---------
clean_dataframe        : Normalise column names, fill missing values, clip outliers
encode_categoricals    : LabelEncode categorical columns in-place
build_tabular_features : Select + scale numeric & encoded categorical features
save_preprocessor      : Persist encoders + scaler to disk
load_preprocessor      : Restore encoders + scaler from disk
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# ── COLUMN CLEANING ──────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame,
                    imputer_strategy: str = "median",
                    clip_outliers: bool = True,
                    label_col: str = "target") -> tuple[pd.DataFrame, dict]:
    """
    Normalise a raw ISIC metadata DataFrame.

    Steps
    -----
    1. Strip & lower-case column names
    2. Fill missing values (numeric → median, categorical → mode)
    3. Clip numeric outliers via IQR method (preserves minority class column)

    Returns
    -------
    (df_clean, report)  where report is a dict with before/after stats.
    """
    report: dict = {}
    df = df.copy()

    # 1. Normalise column names
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r"\s+", "_", regex=True)
                    .str.replace(r"[^a-z0-9_]", "_", regex=True)
                    .str.replace(r"_+", "_", regex=True)
                    .str.strip("_"))

    report["initial_shape"] = df.shape

    # 2. Fill missing values
    missing_before = int(df.isnull().sum().sum())
    report["missing_before"] = missing_before

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object"]).columns.tolist()

    if numeric_cols:
        imputer = SimpleImputer(strategy=imputer_strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    for col in cat_cols_all:
        mode_val = df[col].mode()
        fill = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
        df[col].fillna(fill, inplace=True)

    report["missing_after"] = int(df.isnull().sum().sum())

    # 3. Clip numeric outliers (IQR)
    outlier_report: dict = {}
    if clip_outliers:
        safe_numeric = [c for c in numeric_cols if c != label_col]
        for col in safe_numeric[:50]:   # limit to first 50 for speed
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_outliers = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_outliers > 0:
                outlier_report[col] = {
                    "count": n_outliers,
                    "pct": round(n_outliers / len(df) * 100, 2),
                }
                df[col] = df[col].clip(lo, hi)

    report["outliers"] = outlier_report
    report["final_shape"] = df.shape

    print(f"[clean_dataframe] {report['initial_shape']} → {report['final_shape']}  "
          f"| missing: {missing_before} → {report['missing_after']}  "
          f"| outlier cols clipped: {len(outlier_report)}")
    return df, report


# ── CATEGORICAL ENCODING ─────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame,
                         cat_cols: list[str],
                         encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.

    Parameters
    ----------
    df        : DataFrame (already cleaned)
    cat_cols  : e.g. ['sex', 'anatom_site_general']
    encoders  : pass existing encoders to transform (instead of fit+transform)

    Returns
    -------
    (df_encoded, encoders_dict)
    """
    df = df.copy()
    fit_new = encoders is None
    encoders = encoders or {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

    return df, encoders


# ── FEATURE MATRIX BUILDER ───────────────────────────────────────────────────

def build_tabular_features(df: pd.DataFrame,
                            exclude_cols: list[str],
                            cat_cols: list[str],
                            scaler: StandardScaler | MinMaxScaler | None = None,
                            fit_scaler: bool = True) -> tuple[np.ndarray, list, object]:
    """
    Build the numeric feature matrix from a (cleaned + encoded) DataFrame.

    Parameters
    ----------
    df           : cleaned & encoded DataFrame
    exclude_cols : columns to drop (label, id, etc.)
    cat_cols     : categorical cols already encoded — include in features
    scaler       : pass a fitted scaler to transform (None to create new)
    fit_scaler   : fit the scaler on this data (False = transform only)

    Returns
    -------
    (X_tabular np.ndarray, feature_names list, fitted_scaler)
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Add encoded categorical columns if not already present
    for col in cat_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

    X = df[feature_cols].values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    print(f"[build_tabular_features] Feature matrix: {X.shape}  "
          f"({len(feature_cols)} features)")
    return X.astype(np.float32), feature_cols, scaler


# ── PERSISTENCE ──────────────────────────────────────────────────────────────

def save_preprocessor(artifacts_dir: str,
                       encoders: dict,
                       scaler,
                       label_encoder: LabelEncoder,
                       feature_names: list) -> None:
    """Save all preprocessing artifacts to disk."""
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(os.path.join(artifacts_dir, "cat_encoders.pkl"),   "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(artifacts_dir, "scaler.pkl"),          "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(artifacts_dir, "label_encoder.pkl"),   "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(artifacts_dir, "feature_names.pkl"),   "wb") as f:
        pickle.dump(feature_names, f)

    print(f"[save_preprocessor] Artifacts saved to '{artifacts_dir}'")


def load_preprocessor(artifacts_dir: str) -> dict:
    """
    Load preprocessing artifacts from disk.

    Returns dict with keys: encoders, scaler, label_encoder, feature_names.
    """
    def _load(name):
        with open(os.path.join(artifacts_dir, name), "rb") as f:
            return pickle.load(f)

    return {
        "encoders":      _load("cat_encoders.pkl"),
        "scaler":        _load("scaler.pkl"),
        "label_encoder": _load("label_encoder.pkl"),
        "feature_names": _load("feature_names.pkl"),
    }
