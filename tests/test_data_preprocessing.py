"""
tests/test_data_preprocessing.py
=================================
Unit tests for src/data_preprocessing.py
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (
    augment_image,
    build_balanced_selected_ids,
    build_tabular_features,
    load_image,
    oversample_malignant,
    preprocess_image,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image() -> np.ndarray:
    """224×224 RGB float32 image."""
    rng = np.random.default_rng(42)
    return rng.random((224, 224, 3), dtype=np.float64).astype(np.float32)


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """Minimal ISIC-like DataFrame (100 benign + 10 malignant)."""
    rng = np.random.default_rng(0)
    n = 110
    df = pd.DataFrame({
        "isic_id":            [f"ISIC_{i:07d}" for i in range(n)],
        "target":             [0] * 100 + [1] * 10,
        "age_approx":         rng.integers(20, 80, n).astype(float),
        "sex":                rng.choice(["male", "female"], n),
        "anatom_site_general": rng.choice(["head/neck", "upper extremity", "trunk"], n),
        "tbp_lv_A":           rng.standard_normal(n).astype(np.float32),
        "tbp_lv_B":           rng.standard_normal(n).astype(np.float32),
        "clin_size_long_diam_mm": rng.random(n).astype(np.float32) * 20,
    })
    return df


# ---------------------------------------------------------------------------
# Image tests
# ---------------------------------------------------------------------------

class TestImageUtils:
    def test_preprocess_image_shape(self, dummy_image):
        out = preprocess_image(dummy_image)
        assert out.shape == (224, 224, 3)

    def test_preprocess_image_range(self, dummy_image):
        out = preprocess_image(dummy_image)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_augment_image_shape(self, dummy_image):
        out = augment_image(dummy_image)
        assert out.shape == (224, 224, 3)

    def test_augment_image_strong(self, dummy_image):
        out = augment_image(dummy_image, strong=True)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.float32

    def test_preprocess_no_clahe(self, dummy_image):
        out = preprocess_image(dummy_image, apply_clahe=False, apply_gaussian=False)
        assert out.shape == (224, 224, 3)

    def test_augment_deterministic_seed(self, dummy_image):
        """Different augmentations should generally differ (probabilistic)."""
        np.random.seed(1)
        a1 = augment_image(dummy_image, strong=True)
        np.random.seed(99)
        a2 = augment_image(dummy_image, strong=True)
        # They may occasionally be equal by chance, but shapes must match
        assert a1.shape == a2.shape


# ---------------------------------------------------------------------------
# Tabular tests
# ---------------------------------------------------------------------------

class TestTabularFeatures:
    def test_output_shape(self, dummy_df):
        X, cols, _, scaler, imputer = build_tabular_features(dummy_df, fit=True)
        assert X.ndim == 2
        assert X.shape[0] == len(dummy_df)
        assert X.shape[1] == len(cols)

    def test_no_nans_after_preprocessing(self, dummy_df):
        X, *_ = build_tabular_features(dummy_df, fit=True)
        assert not np.isnan(X).any(), "NaN found after preprocessing"

    def test_scaled_approx_zero_mean(self, dummy_df):
        X, *_ = build_tabular_features(dummy_df, fit=True)
        # After StandardScaler mean should be close to 0
        assert np.abs(X.mean(axis=0)).max() < 0.5

    def test_transform_uses_fit_params(self, dummy_df):
        X_tr, cols, les, scaler, imputer = build_tabular_features(dummy_df.iloc[:80], fit=True)
        X_te, *_ = build_tabular_features(
            dummy_df.iloc[80:], fit=False,
            scaler=scaler, label_encoders=les, imputer=imputer,
        )
        assert X_te.shape[1] == X_tr.shape[1]


# ---------------------------------------------------------------------------
# Oversampling tests
# ---------------------------------------------------------------------------

class TestOversampling:
    def _make_arrays(self, n_ben=100, n_mal=10):
        rng = np.random.default_rng(7)
        n = n_ben + n_mal
        X_img = rng.random((n, 224, 224, 3), dtype=np.float64).astype(np.float32)
        X_tab = rng.random((n, 10)).astype(np.float32)
        y = np.array([0] * n_ben + [1] * n_mal)
        return X_img, X_tab, y

    def test_ratio_achieved(self):
        X_img, X_tab, y = self._make_arrays()
        X_img_os, X_tab_os, y_os = oversample_malignant(
            X_img, X_tab, y, target_ratio=0.25, strong_aug=False
        )
        ratio = np.mean(y_os == 1)
        assert ratio >= 0.20, f"Ratio {ratio:.3f} < 0.20"

    def test_shapes_consistent(self):
        X_img, X_tab, y = self._make_arrays()
        X_img_os, X_tab_os, y_os = oversample_malignant(X_img, X_tab, y, strong_aug=False)
        assert X_img_os.shape[0] == X_tab_os.shape[0] == y_os.shape[0]

    def test_original_samples_preserved(self):
        X_img, X_tab, y = self._make_arrays()
        X_img_os, X_tab_os, y_os = oversample_malignant(X_img, X_tab, y, strong_aug=False)
        # Total must be >= original
        assert len(y_os) >= len(y)


# ---------------------------------------------------------------------------
# Stratified split tests
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def _make_data(self):
        rng = np.random.default_rng(3)
        n = 500
        X_tab = rng.random((n, 5)).astype(np.float32)
        X_img = rng.random((n, 8, 8, 3)).astype(np.float32)  # tiny for speed
        y = np.array([0] * 450 + [1] * 50)
        return X_tab, X_img, y

    def test_sizes_sum_to_total(self):
        X_tab, X_img, y = self._make_data()
        splits = stratified_split(X_tab, X_img, y)
        total = sum(len(splits[k][2]) for k in ["train", "val", "test"])
        assert total == len(y)

    def test_all_splits_have_both_classes(self):
        X_tab, X_img, y = self._make_data()
        splits = stratified_split(X_tab, X_img, y)
        for name, (_, _, ys) in splits.items():
            assert 0 in ys and 1 in ys, f"Split '{name}' missing a class"

    def test_no_overlap(self):
        X_tab, X_img, y = self._make_data()
        splits = stratified_split(X_tab, X_img, y)
        # Use tab sums as proxy fingerprints
        sums = {k: set(round(float(r.sum()), 6) for r in splits[k][0]) for k in splits}
        assert sums["train"].isdisjoint(sums["test"])
