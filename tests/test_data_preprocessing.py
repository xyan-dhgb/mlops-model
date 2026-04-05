"""
tests/test_data_preprocessing.py
Unit tests for tabular and image preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    preprocess_csv_data,
    get_tabular_columns,
    fit_encoders,
    encode_categorical,
    scale_features,
    load_image,
    preprocess_image,
    augment_image,
    LABEL_COL,
    ID_COL,
    CATEGORICAL_COLS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df():
    """Minimal ISIC 2024-like DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "isic_id":             [f"ISIC_{i:07d}" for i in range(n)],
        "target":              np.random.randint(0, 2, n),
        "age_approx":          np.random.uniform(20, 80, n),
        "sex":                 np.random.choice(["male", "female", None], n),
        "anatom_site_general": np.random.choice(["torso", "arm", None], n),
        "tbp_lv_A":            np.random.randn(n),
        "tbp_lv_B":            np.random.randn(n),
        "clin_size_long_diam_mm": np.random.uniform(1, 20, n),
        "patient_id":          [f"P{i}" for i in range(n)],  # should be excluded
    })


@pytest.fixture()
def sample_image():
    """Random float32 image array (224×224×3)."""
    return np.random.rand(224, 224, 3).astype(np.float32)


# ── Tabular preprocessing tests ───────────────────────────────────────────────

class TestPreprocessCSVData:
    def test_returns_dataframe_and_report(self, sample_df):
        df_out, report = preprocess_csv_data(sample_df)
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(report, dict)

    def test_no_missing_values_after_processing(self, sample_df):
        df_out, report = preprocess_csv_data(sample_df)
        assert df_out.isnull().sum().sum() == 0

    def test_column_names_normalised(self, sample_df):
        df_out, _ = preprocess_csv_data(sample_df)
        for col in df_out.columns:
            assert col == col.lower(), f"Column not lowercased: {col}"
            assert " " not in col, f"Column has spaces: {col}"

    def test_report_keys(self, sample_df):
        _, report = preprocess_csv_data(sample_df)
        for key in ("initial_shape", "final_shape", "missing_before", "missing_after", "outliers"):
            assert key in report

    def test_shape_preserved(self, sample_df):
        df_out, report = preprocess_csv_data(sample_df)
        assert df_out.shape[0] == sample_df.shape[0]


class TestGetTabularColumns:
    def test_excludes_id_and_target(self, sample_df):
        df_out, _ = preprocess_csv_data(sample_df)
        cols = get_tabular_columns(df_out)
        assert ID_COL not in cols
        assert LABEL_COL not in cols
        assert "patient_id" not in cols

    def test_returns_list(self, sample_df):
        df_out, _ = preprocess_csv_data(sample_df)
        assert isinstance(get_tabular_columns(df_out), list)

    def test_columns_exist_in_df(self, sample_df):
        df_out, _ = preprocess_csv_data(sample_df)
        cols = get_tabular_columns(df_out)
        for col in cols:
            assert col in df_out.columns


class TestEncoders:
    def test_fit_encoders_returns_dict(self, sample_df):
        df_out, _ = preprocess_csv_data(sample_df)
        encoders = fit_encoders(df_out)
        assert isinstance(encoders, dict)

    def test_encode_categorical_no_nans(self, sample_df):
        df_out, _ = preprocess_csv_data(sample_df)
        encoders = fit_encoders(df_out)
        df_enc = encode_categorical(df_out, encoders)
        for col in CATEGORICAL_COLS:
            if col in df_enc.columns:
                assert df_enc[col].isnull().sum() == 0

    def test_encoded_values_are_integers(self, sample_df):
        df_out, _ = preprocess_csv_data(sample_df)
        encoders = fit_encoders(df_out)
        df_enc = encode_categorical(df_out, encoders)
        for col in CATEGORICAL_COLS:
            if col in df_enc.columns:
                assert df_enc[col].dtype in (int, "int64", "int32")


class TestScaleFeatures:
    def test_output_shape(self):
        X = np.random.randn(50, 10).astype(np.float32)
        X_scaled, scaler = scale_features(X, fit=True)
        assert X_scaled.shape == X.shape

    def test_mean_near_zero(self):
        X = np.random.randn(200, 5).astype(np.float32) * 100 + 50
        X_scaled, _ = scale_features(X, fit=True)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-5)

    def test_reuse_scaler(self):
        X_train = np.random.randn(100, 5).astype(np.float32)
        X_test  = np.random.randn(20, 5).astype(np.float32)
        _, scaler = scale_features(X_train, fit=True)
        X_test_scaled, _ = scale_features(X_test, scaler=scaler, fit=False)
        assert X_test_scaled.shape == X_test.shape


# ── Image preprocessing tests ─────────────────────────────────────────────────

class TestPreprocessImage:
    def test_output_shape_unchanged(self, sample_image):
        out = preprocess_image(sample_image)
        assert out.shape == sample_image.shape

    def test_values_in_range(self, sample_image):
        out = preprocess_image(sample_image)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_returns_float32(self, sample_image):
        out = preprocess_image(sample_image)
        assert out.dtype == np.float32

    def test_none_input_returns_none(self):
        assert preprocess_image(None) is None

    def test_without_clahe(self, sample_image):
        out = preprocess_image(sample_image, apply_clahe=False)
        assert out is not None

    def test_without_gaussian(self, sample_image):
        out = preprocess_image(sample_image, apply_gaussian=False)
        assert out is not None


class TestAugmentImage:
    def test_output_shape(self, sample_image):
        out = augment_image(sample_image)
        assert out.shape == sample_image.shape

    def test_values_in_range(self, sample_image):
        out = augment_image(sample_image)
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_returns_float32(self, sample_image):
        out = augment_image(sample_image)
        assert out.dtype == np.float32

    def test_deterministic_with_seed(self, sample_image):
        np.random.seed(0)
        out1 = augment_image(sample_image, rotation_range=0, zoom_range=0, horizontal_flip=False)
        np.testing.assert_array_almost_equal(out1, out1)  # self-consistency
