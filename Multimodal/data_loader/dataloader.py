"""
DataLoader for ISIC 2024 multimodal training.
Produces paired (tabular, image, label) arrays ready for model.fit().
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from preprocessing.image_preprocessing import load_image, preprocess_image
from preprocessing.tabular_preprocessing import (
    get_tabular_columns,
    fit_encoders,
    encode_categorical,
    scale_features,
    LABEL_COL,
    ID_COL,
    CATEGORICAL_COLS,
)


def prepare_multimodal_data(
    df: pd.DataFrame,
    image_dir: str,
    target_size: tuple = (224, 224),
    is_training: bool = True
) -> tuple:
    """
    Build aligned tabular and image arrays from ISIC 2024 metadata.

    Args:
        df:           Filtered DataFrame (rows with matching images).
        image_dir:    Directory containing extracted .jpg files.
        target_size:  Image resize target.
        is_training:  If True, apply augmentation-friendly preprocessing.

    Returns:
        (X_tabular, X_image, y, encoders, tabular_cols)
        All arrays are float32 numpy arrays.
    """
    tabular_cols = get_tabular_columns(df)

    # Encode categoricals
    encoders = fit_encoders(df, CATEGORICAL_COLS)
    df_enc = encode_categorical(df, encoders)

    # Add encoded categorical cols to feature set
    for col in CATEGORICAL_COLS:
        if col in df_enc.columns and col not in tabular_cols:
            tabular_cols.append(col)

    # Impute missing numerics
    imputer = SimpleImputer(strategy="median")
    df_enc[tabular_cols] = imputer.fit_transform(df_enc[tabular_cols])

    X_tabular, X_image, y = [], [], []

    label_enc = LabelEncoder()
    y_all = label_enc.fit_transform(df_enc[LABEL_COL])

    for idx, (_, row) in enumerate(df_enc.iterrows()):
        isic_id = row[ID_COL]
        img_path = os.path.join(image_dir, f"{isic_id}.jpg")

        if not os.path.exists(img_path):
            continue

        img = load_image(img_path, target_size=target_size)
        if img is None:
            continue
        img = preprocess_image(img, apply_clahe=True, apply_gaussian=True)
        if img is None:
            continue

        tab = row[tabular_cols].values.astype(np.float32)

        X_tabular.append(tab)
        X_image.append(img)
        y.append(y_all[idx])

    X_tabular = np.array(X_tabular, dtype=np.float32)
    X_image = np.array(X_image, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[DataLoader] Tabular: {X_tabular.shape}, Image: {X_image.shape}, Labels: {y.shape}")
    print(f"  Benign: {int((y == 0).sum())}, Malignant: {int((y == 1).sum())}")

    return X_tabular, X_image, y, encoders, tabular_cols


def build_dataloaders(
    df: pd.DataFrame,
    image_dir: str,
    target_size: tuple = (224, 224),
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Full pipeline: prepare data → stratified split into train / val / test.

    Args:
        df:           Filtered metadata DataFrame.
        image_dir:    Path to extracted images.
        target_size:  Resize dimensions.
        test_size:    Fraction for test split.
        val_size:     Fraction of train+val used for validation.
        random_state: Reproducibility seed.

    Returns:
        Dict with keys: train, val, test (each a dict of X_tab, X_img, y),
        plus 'encoders' and 'tabular_cols'.
    """
    X_tab, X_img, y, encoders, tabular_cols = prepare_multimodal_data(
        df, image_dir, target_size
    )

    # Train+val / test
    X_img_tv, X_img_test, X_tab_tv, X_tab_test, y_tv, y_test = train_test_split(
        X_img, X_tab, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train / val
    X_img_train, X_img_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
        X_img_tv, X_tab_tv, y_tv, test_size=val_size, random_state=random_state, stratify=y_tv
    )

    splits = {
        "train": {"X_tab": X_tab_train, "X_img": X_img_train, "y": y_train},
        "val":   {"X_tab": X_tab_val,   "X_img": X_img_val,   "y": y_val},
        "test":  {"X_tab": X_tab_test,  "X_img": X_img_test,  "y": y_test},
        "encoders":     encoders,
        "tabular_cols": tabular_cols,
    }

    for split_name in ("train", "val", "test"):
        s = splits[split_name]
        print(f"[{split_name}] tab={s['X_tab'].shape}, img={s['X_img'].shape}, "
              f"benign={int((s['y'] == 0).sum())}, malignant={int((s['y'] == 1).sum())}")

    return splits
