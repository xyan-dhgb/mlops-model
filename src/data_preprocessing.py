"""
src/data_preprocessing.py
Thin public API consumed by tests and the run_train script.
Delegates to Multimodal/preprocessing/* for the actual logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Multimodal.preprocessing.tabular_preprocessing import (
    preprocess_csv_data,
    get_tabular_columns,
    fit_encoders,
    encode_categorical,
    scale_features,
    save_preprocessor,
    load_preprocessor,
    LABEL_COL,
    ID_COL,
    CATEGORICAL_COLS,
)
from Multimodal.preprocessing.image_preprocessing import (
    load_image,
    preprocess_image,
    augment_image,
    extract_images_from_hdf5,
)
from Multimodal.data_loader.dataloader import build_dataloaders


__all__ = [
    # tabular
    "preprocess_csv_data",
    "get_tabular_columns",
    "fit_encoders",
    "encode_categorical",
    "scale_features",
    "save_preprocessor",
    "load_preprocessor",
    "LABEL_COL",
    "ID_COL",
    "CATEGORICAL_COLS",
    # image
    "load_image",
    "preprocess_image",
    "augment_image",
    "extract_images_from_hdf5",
    # pipeline
    "build_dataloaders",
]
