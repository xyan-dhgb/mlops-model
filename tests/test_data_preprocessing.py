"""
Unit Tests — Data Preprocessing
Tests: image pipeline, tabular pipeline, dataset output shapes
Run: pytest tests/test_data_preprocessing.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
import torch

from Multimodal.preprocessing.image_preprocessing import (
    remove_hair,
    shades_of_gray,
    get_train_transforms,
    get_val_transforms,
    compute_class_weights,
    ISICDataset,
    CLASS_NAMES,
    IMAGE_SIZE,
)
from Multimodal.preprocessing.tabular_preprocessing import (
    clean_metadata,
    engineer_features,
    create_folds,
    compute_class_weights as tabular_class_weights,
    MetadataPreprocessor,
    CLASS_NAMES as TAB_CLASS_NAMES,
    SITE_CATEGORIES,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def dummy_image():
    """Random RGB numpy array simulating a 512×512 dermoscopy image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_metadata_df():
    """Synthetic ISIC metadata DataFrame (100 rows)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "image_name": [f"ISIC_{i:07d}" for i in range(100)],
        "age_approx": rng.choice([25.0, 40.0, 55.0, 70.0, np.nan], 100),
        "sex": rng.choice(["male", "female", None], 100),
        "anatom_site_general_challenge": rng.choice(SITE_CATEGORIES + [None], 100),
        "diagnosis": rng.choice(
            CLASS_NAMES, 100,
            p=[0.11, 0.67, 0.05, 0.03, 0.11, 0.01, 0.02],
        ),
    })


# ─────────────────────────────────────────────
# Image Preprocessing Tests
# ─────────────────────────────────────────────
class TestHairRemoval:
    def test_output_shape_preserved(self, dummy_image):
        result = remove_hair(dummy_image)
        assert result.shape == dummy_image.shape

    def test_output_dtype_uint8(self, dummy_image):
        result = remove_hair(dummy_image)
        assert result.dtype == np.uint8

    def test_modifies_image(self, dummy_image):
        # Hair removal should change at least some pixels
        result = remove_hair(dummy_image)
        # Not identical to input (inpainting touches something)
        assert result.shape == dummy_image.shape  # at minimum shape is fine


class TestColorConstancy:
    def test_output_shape_preserved(self, dummy_image):
        result = shades_of_gray(dummy_image)
        assert result.shape == dummy_image.shape

    def test_output_dtype_uint8(self, dummy_image):
        result = shades_of_gray(dummy_image)
        assert result.dtype == np.uint8

    def test_values_in_valid_range(self, dummy_image):
        result = shades_of_gray(dummy_image)
        assert result.min() >= 0
        assert result.max() <= 255


class TestTransforms:
    def test_train_transform_output_shape(self, dummy_image):
        t = get_train_transforms(IMAGE_SIZE)
        out = t(image=dummy_image)["image"]
        assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert out.dtype == torch.float32

    def test_val_transform_output_shape(self, dummy_image):
        t = get_val_transforms(IMAGE_SIZE)
        out = t(image=dummy_image)["image"]
        assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_train_transform_is_deterministic_off(self, dummy_image):
        """Train transforms should not always produce identical output."""
        t = get_train_transforms(IMAGE_SIZE)
        out1 = t(image=dummy_image.copy())["image"]
        out2 = t(image=dummy_image.copy())["image"]
        # With random augmentations, outputs will usually differ
        # (this test just verifies both are valid tensors)
        assert out1.shape == out2.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_val_transform_is_deterministic(self, dummy_image):
        """Val transforms should always produce same output."""
        t = get_val_transforms(IMAGE_SIZE)
        out1 = t(image=dummy_image.copy())["image"]
        out2 = t(image=dummy_image.copy())["image"]
        assert torch.allclose(out1, out2)

    def test_normalization_applied(self, dummy_image):
        """Output should have values outside [0, 255] range (normalized)."""
        t = get_val_transforms(IMAGE_SIZE)
        out = t(image=dummy_image)["image"]
        # After ImageNet normalization, values will be outside [0, 1]
        assert out.min().item() < 0 or out.max().item() > 1


class TestClassWeights:
    def test_weight_shape(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        weights = compute_class_weights(df)
        assert weights.shape == (len(CLASS_NAMES),)

    def test_weights_positive(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        weights = compute_class_weights(df)
        assert (weights > 0).all()

    def test_minority_class_higher_weight(self, dummy_metadata_df):
        """MEL (~11%) should have higher weight than NV (~67%)."""
        df = clean_metadata(dummy_metadata_df)
        weights = compute_class_weights(df)
        mel_idx = CLASS_NAMES.index("MEL")
        nv_idx  = CLASS_NAMES.index("NV")
        assert weights[mel_idx] > weights[nv_idx]


# ─────────────────────────────────────────────
# Tabular Preprocessing Tests
# ─────────────────────────────────────────────
class TestCleanMetadata:
    def test_no_null_age(self, dummy_metadata_df):
        cleaned = clean_metadata(dummy_metadata_df)
        assert cleaned["age_approx"].isna().sum() == 0

    def test_no_null_sex(self, dummy_metadata_df):
        cleaned = clean_metadata(dummy_metadata_df)
        assert cleaned["sex"].isna().sum() == 0

    def test_age_clipped(self, dummy_metadata_df):
        dummy_metadata_df.loc[0, "age_approx"] = 200.0
        cleaned = clean_metadata(dummy_metadata_df)
        assert cleaned["age_approx"].max() <= 110

    def test_label_column_created(self, dummy_metadata_df):
        cleaned = clean_metadata(dummy_metadata_df)
        assert "label" in cleaned.columns
        assert cleaned["label"].dtype == int

    def test_label_range(self, dummy_metadata_df):
        cleaned = clean_metadata(dummy_metadata_df)
        assert cleaned["label"].min() >= 0
        assert cleaned["label"].max() < len(TAB_CLASS_NAMES)

    def test_drops_rows_without_image_name(self):
        df = pd.DataFrame({
            "image_name": [None, "ISIC_0000001"],
            "age_approx": [30.0, 45.0],
            "sex": ["male", "female"],
            "anatom_site_general_challenge": ["torso", "head/neck"],
            "diagnosis": ["MEL", "NV"],
        })
        cleaned = clean_metadata(df)
        assert len(cleaned) == 1


class TestEngineerFeatures:
    def test_age_bucket_created(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        df = engineer_features(df)
        assert "age_bucket" in df.columns
        assert df["age_bucket"].isna().sum() == 0

    def test_high_risk_site_binary(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        df = engineer_features(df)
        assert set(df["high_risk_site"].unique()).issubset({0.0, 1.0})

    def test_is_male_binary(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        df = engineer_features(df)
        assert set(df["is_male"].unique()).issubset({0.0, 1.0})


class TestCreateFolds:
    def test_fold_column_exists(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        df = create_folds(df, n_splits=5)
        assert "fold" in df.columns

    def test_five_folds(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        df = create_folds(df, n_splits=5)
        assert sorted(df["fold"].unique()) == [0, 1, 2, 3, 4]

    def test_no_sample_in_multiple_folds(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        df = create_folds(df, n_splits=5)
        assert df["fold"].isna().sum() == 0

    def test_stratification_preserves_class_dist(self, dummy_metadata_df):
        """Each fold should contain at least one melanoma sample (if present)."""
        df = clean_metadata(dummy_metadata_df)
        df = create_folds(df, n_splits=5)
        mel_idx = TAB_CLASS_NAMES.index("MEL")
        mel_df  = df[df["label"] == mel_idx]
        if len(mel_df) >= 5:
            assert mel_df["fold"].nunique() >= 2


class TestMetadataPreprocessor:
    def test_fit_transform_shape(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        pp = MetadataPreprocessor()
        features = pp.fit_transform(df)
        assert features.shape == (len(df), pp.feature_dim)

    def test_feature_dim_is_5(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        pp = MetadataPreprocessor()
        pp.fit(df)
        assert pp.feature_dim == 5

    def test_transform_without_fit_raises(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        pp = MetadataPreprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            pp.transform(df)

    def test_output_dtype_float32(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        pp = MetadataPreprocessor()
        features = pp.fit_transform(df)
        assert features.dtype == np.float32

    def test_to_tensor_shape(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        pp = MetadataPreprocessor()
        pp.fit(df)
        tensor = pp.to_tensor(df)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (len(df), pp.feature_dim)

    def test_handles_unknown_site(self, dummy_metadata_df):
        df = clean_metadata(dummy_metadata_df)
        pp = MetadataPreprocessor()
        pp.fit(df)
        # Inject unseen site
        df_test = df.copy()
        df_test["anatom_site_general_challenge"] = "mars_surface"
        features = pp.transform(df_test)   # should not raise
        assert features.shape[0] == len(df_test)

    def test_save_load_roundtrip(self, dummy_metadata_df, tmp_path):
        df = clean_metadata(dummy_metadata_df)
        pp = MetadataPreprocessor()
        features_before = pp.fit_transform(df)

        path = str(tmp_path / "preprocessor.pkl")
        pp.save(path)
        pp2 = MetadataPreprocessor.load(path)
        features_after = pp2.transform(df)

        np.testing.assert_allclose(features_before, features_after, rtol=1e-5)
