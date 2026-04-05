"""
tests/conftest.py
Shared pytest configuration and fixtures.
"""

import sys
import os

# Ensure project root is on PYTHONPATH so all imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def isic_like_df():
    """
    Session-scoped ISIC 2024-like DataFrame with 200 rows.
    Mirrors the column layout of the real train-metadata.csv.
    """
    np.random.seed(99)
    n = 200
    return pd.DataFrame({
        "isic_id":                   [f"ISIC_{i:07d}" for i in range(n)],
        "target":                     np.random.randint(0, 2, n),
        "age_approx":                 np.random.uniform(20, 80, n),
        "sex":                        np.random.choice(["male", "female"], n),
        "anatom_site_general":        np.random.choice(["torso", "lower extremity", "head/neck"], n),
        "clin_size_long_diam_mm":     np.random.uniform(1.0, 25.0, n),
        "tbp_lv_A":                   np.random.randn(n),
        "tbp_lv_B":                   np.random.randn(n),
        "tbp_lv_Aext":                np.random.randn(n),
        "tbp_lv_Bext":                np.random.randn(n),
        "tbp_lv_C":                   np.random.randn(n),
        "tbp_lv_Cext":                np.random.randn(n),
        "tbp_lv_H":                   np.random.randn(n),
        "tbp_lv_Hext":                np.random.randn(n),
        "tbp_lv_L":                   np.random.randn(n),
        "tbp_lv_areaMM2":             np.random.uniform(1, 200, n),
        "tbp_lv_area_perim_ratio":    np.random.uniform(0.1, 5.0, n),
        "tbp_lv_color_std_mean":      np.random.uniform(0, 1, n),
        "tbp_lv_deltaA":              np.random.randn(n),
        "tbp_lv_deltaB":              np.random.randn(n),
        "tbp_lv_deltaL":              np.random.randn(n),
        "tbp_lv_deltaLBnorm":         np.random.randn(n),
        "tbp_lv_eccentricity":        np.random.uniform(0, 1, n),
        "tbp_lv_minorAxisMM":         np.random.uniform(0.5, 10, n),
        "tbp_lv_nevi_confidence":     np.random.uniform(0, 1, n),
        "tbp_lv_norm_border":         np.random.uniform(0, 1, n),
        "tbp_lv_norm_color":          np.random.uniform(0, 1, n),
        "tbp_lv_perimeterMM":         np.random.uniform(5, 100, n),
        "tbp_lv_radial_color_std_max":np.random.uniform(0, 1, n),
        "tbp_lv_stdL":                np.random.uniform(0, 50, n),
        "tbp_lv_stdLExt":             np.random.uniform(0, 50, n),
        "tbp_lv_symm_2axis":          np.random.uniform(0, 1, n),
        "tbp_lv_symm_2axis_angle":    np.random.uniform(0, 180, n),
        "tbp_lv_x":                   np.random.randn(n),
        "tbp_lv_y":                   np.random.randn(n),
        "tbp_lv_z":                   np.random.randn(n),
        # Columns that should be excluded from features
        "patient_id":                 [f"P{i:05d}" for i in range(n)],
        "attribution":                ["ISIC"] * n,
        "copyright_license":          ["CC-BY"] * n,
        "image_type":                 ["TBP tile: close-up"] * n,
        "iddx_full":                  [None] * n,
        "iddx_1":                     [None] * n,
        "mel_mitotic_index":          [None] * n,
        "mel_thick_mm":               [None] * n,
        "lesion_id":                  [None] * n,
    })


@pytest.fixture(scope="session")
def tiny_image():
    """64×64×3 float32 image for fast unit tests."""
    np.random.seed(42)
    return np.random.rand(64, 64, 3).astype(np.float32)


@pytest.fixture(scope="session")
def tiny_image_batch():
    """Batch of 8 tiny images."""
    np.random.seed(42)
    return np.random.rand(8, 64, 64, 3).astype(np.float32)
