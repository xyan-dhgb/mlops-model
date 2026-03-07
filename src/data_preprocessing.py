"""
Data preprocessing module for Skin Cancer Multimodal Classification.
Dataset: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
TABULAR_COLS = ["age", "fitspatrick", "diameter_1", "diameter_2"]
TARGET_COL = "diagnostic"
IMG_ID_COL = "img_id"
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# CSV / Tabular
# ─────────────────────────────────────────────
def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load metadata CSV and return a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    df = pd.read_csv(file_path)
    return df


def preprocess_csv_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and normalise the raw metadata DataFrame.

    Returns
    -------
    df_processed : pd.DataFrame
    report       : dict  – preprocessing statistics
    """
    df_processed = df.copy()
    report: dict = {}

    report["initial_shape"] = df_processed.shape

    # Normalise column names
    df_processed.columns = (
        df_processed.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    # Missing-value stats before imputation
    report["missing_before"] = int(df_processed.isnull().sum().sum())

    # Impute numeric columns with median
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])

    # Fill categorical NaN with mode
    cat_cols = df_processed.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if df_processed[col].isnull().any():
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    report["missing_after"] = int(df_processed.isnull().sum().sum())
    report["final_shape"] = df_processed.shape
    return df_processed, report


def encode_tabular_features(
    df: pd.DataFrame,
    tabular_cols: list[str] | None = None,
    fit: bool = True,
    scaler: StandardScaler | None = None,
    label_encoder: LabelEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Scale numeric features and encode the target label.

    Returns
    -------
    X_tabular     : np.ndarray – scaled features
    y_encoded     : np.ndarray – integer labels
    scaler        : fitted StandardScaler
    label_encoder : fitted LabelEncoder
    """
    cols = tabular_cols or TABULAR_COLS

    # Keep only columns that exist in df
    cols = [c for c in cols if c in df.columns]

    X = df[cols].values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[TARGET_COL].astype(str))
    else:
        assert scaler is not None and label_encoder is not None
        X = scaler.transform(X)
        y = label_encoder.transform(df[TARGET_COL].astype(str))

    return X, y, scaler, label_encoder


# ─────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────
def load_image(image_path: str, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load a single image, resize, and normalise to [0, 1]."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB").resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0


def prepare_multimodal_data(
    df: pd.DataFrame,
    image_dir: str,
    target_size: tuple[int, int] = IMAGE_SIZE,
    tabular_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Build arrays of (tabular features, images, labels) from the DataFrame.

    Rows whose image file is missing are silently skipped.
    """
    cols = tabular_cols or TABULAR_COLS
    cols = [c for c in cols if c in df.columns]

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET_COL].astype(str))

    X_tabular, X_image, y = [], [], []

    # Simple median imputation for tabular
    imputer = SimpleImputer(strategy="median")
    tab_data = imputer.fit_transform(df[cols].values.astype(np.float32))

    scaler = StandardScaler()
    tab_data = scaler.fit_transform(tab_data)

    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, str(row[IMG_ID_COL]))
        if not os.path.exists(img_path):
            continue
        try:
            img = load_image(img_path, target_size)
        except Exception:
            continue

        X_tabular.append(tab_data[idx])
        X_image.append(img)
        y.append(label_encoder.transform([str(row[TARGET_COL])])[0])

    return (
        np.array(X_tabular, dtype=np.float32),
        np.array(X_image, dtype=np.float32),
        np.array(y, dtype=np.int32),
        label_encoder,
    )


def split_dataset(
    X_tabular: np.ndarray,
    X_image: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """
    Split data into train / val / test sets.

    Returns a dict with keys:
      X_tab_train, X_tab_val, X_tab_test,
      X_img_train, X_img_val, X_img_test,
      y_train, y_val, y_test
    """
    # First split: hold out test set
    X_img_tv, X_img_test, X_tab_tv, X_tab_test, y_tv, y_test = train_test_split(
        X_image, X_tabular, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train / val from remaining
    X_img_train, X_img_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
        X_img_tv, X_tab_tv, y_tv, test_size=val_size, random_state=random_state, stratify=y_tv
    )

    return {
        "X_tab_train": X_tab_train,
        "X_tab_val": X_tab_val,
        "X_tab_test": X_tab_test,
        "X_img_train": X_img_train,
        "X_img_val": X_img_val,
        "X_img_test": X_img_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
