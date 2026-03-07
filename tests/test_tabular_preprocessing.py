"""
tests/unit/test_tabular_preprocessing.py
Unit tests cho TabularPreprocessor
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.preprocessing.tabular_preprocessing import (
    TabularPreprocessor,
    LABEL_MAP,
    TOP_LOCALIZATIONS,
)


# ══════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════
def make_df(n=100, seed=42) -> pd.DataFrame:
    """Tạo DataFrame giả HAM10000 metadata"""
    rng = np.random.default_rng(seed)
    dx_codes = list(LABEL_MAP.keys())
    locs     = TOP_LOCALIZATIONS[:5] + ["neck", "foot"]

    return pd.DataFrame({
        "image_id":     [f"ISIC_{i:07d}" for i in range(n)],
        "age":          rng.choice([*range(20, 80), np.nan], size=n),
        "sex":          rng.choice(["male", "female", "unknown"], size=n),
        "localization": rng.choice(locs, size=n),
        "dx":           rng.choice(dx_codes, size=n),
        "dx_type":      rng.choice(["histo", "follow_up", "consensus", "confocal"], size=n),
    })


@pytest.fixture
def train_df():
    return make_df(n=70, seed=1)


@pytest.fixture
def val_df():
    return make_df(n=15, seed=2)


@pytest.fixture
def test_df():
    return make_df(n=15, seed=3)


@pytest.fixture
def fitted_preprocessor(train_df, val_df, test_df):
    """Preprocessor đã được fit trên train"""
    pp = TabularPreprocessor()
    pp.fit_transform_splits(train_df, val_df, test_df)
    return pp


# ══════════════════════════════════════════════════════════
# TEST: _clean
# ══════════════════════════════════════════════════════════
class TestClean:
    def test_removes_unknown_dx(self):
        """Rows với dx không có trong LABEL_MAP phải bị xóa"""
        pp = TabularPreprocessor()
        df = make_df(n=20)
        df.loc[0, "dx"] = "unknown_code"  # Không có trong LABEL_MAP
        cleaned = pp._clean(df)
        assert "unknown_code" not in cleaned["dx"].values

    def test_sex_normalized_to_lowercase(self):
        pp = TabularPreprocessor()
        df = make_df(n=10)
        df["sex"] = ["Male", "FEMALE", "Unknown", "male", "Female",
                     "MALE", "unknown", "female", "Male", "Female"]
        cleaned = pp._clean(df)
        assert all(s in ["male", "female", "unknown"] for s in cleaned["sex"])

    def test_rare_localization_mapped_to_other(self):
        """Localization không phổ biến → 'other'"""
        pp = TabularPreprocessor()
        df = make_df(n=10)
        df["localization"] = "very_rare_body_part"
        cleaned = pp._clean(df)
        assert (cleaned["localization"] == "other").all()

    def test_label_column_added(self):
        pp = TabularPreprocessor()
        df = make_df(n=20)
        cleaned = pp._clean(df)
        assert "label" in cleaned.columns
        assert all(v in ["ACK","BCC","MEL","NEV","SCC","SEK"]
                   for v in cleaned["label"])

    def test_missing_age_kept_as_nan(self):
        """Missing age không bị loại, sẽ xử lý bằng Imputer sau"""
        pp = TabularPreprocessor()
        df = make_df(n=20)
        df["age"] = np.nan  # Tất cả missing
        cleaned = pp._clean(df)
        assert cleaned["age"].isna().all()
        assert len(cleaned) > 0  # Records không bị loại


# ══════════════════════════════════════════════════════════
# TEST: _encode_features
# ══════════════════════════════════════════════════════════
class TestEncodeFeatures:
    def test_age_scaled_is_numeric(self, train_df):
        pp = TabularPreprocessor()
        clean = pp._clean(train_df)
        encoded = pp._encode_features(clean, fit=True)
        assert pd.api.types.is_float_dtype(encoded["age_scaled"])

    def test_age_imputed_no_nan(self, train_df):
        """Sau impute, không được có NaN trong age_scaled"""
        pp = TabularPreprocessor()
        df = train_df.copy()
        df["age"] = np.nan  # Force tất cả missing
        clean = pp._clean(df)
        encoded = pp._encode_features(clean, fit=True)
        assert not encoded["age_scaled"].isna().any()

    def test_sex_one_hot_sum_is_one(self, train_df):
        """Mỗi row: sex_male + sex_female + sex_unknown == 1"""
        pp = TabularPreprocessor()
        clean = pp._clean(train_df)
        encoded = pp._encode_features(clean, fit=True)
        row_sum = encoded[["sex_male","sex_female","sex_unknown"]].sum(axis=1)
        assert (row_sum == 1).all(), "Sex one-hot sum không bằng 1"

    def test_localization_one_hot_sum_is_one(self, train_df):
        """Mỗi row: tổng loc_* == 1"""
        pp = TabularPreprocessor()
        clean = pp._clean(train_df)
        encoded = pp._encode_features(clean, fit=True)
        loc_cols = [c for c in encoded.columns if c.startswith("loc_")]
        assert len(loc_cols) > 0
        row_sum = encoded[loc_cols].sum(axis=1)
        assert (row_sum == 1).all(), "Localization one-hot sum không bằng 1"

    def test_feature_names_saved_after_fit(self, train_df):
        pp = TabularPreprocessor()
        clean = pp._clean(train_df)
        pp._encode_features(clean, fit=True)
        assert len(pp.feature_names_) > 0
        assert "age_scaled" in pp.feature_names_
        assert "sex_male" in pp.feature_names_

    def test_n_features_is_20(self, train_df):
        """Tổng số features phải là 20"""
        pp = TabularPreprocessor()
        clean = pp._clean(train_df)
        encoded = pp._encode_features(clean, fit=True)
        feature_cols = [c for c in encoded.columns
                        if c not in ["image_id","label","split"]]
        assert len(feature_cols) == 20, f"Expected 20, got {len(feature_cols)}: {feature_cols}"


# ══════════════════════════════════════════════════════════
# TEST: fit_transform_splits (data leakage prevention)
# ══════════════════════════════════════════════════════════
class TestFitTransformSplits:
    def test_scaler_fitted_only_on_train(self, train_df, val_df, test_df):
        """
        QUAN TRỌNG: Scaler phải fit trên train TRƯỚC,
        rồi mới transform val/test với tham số của train
        """
        pp = TabularPreprocessor()
        train_feat, val_feat, test_feat = pp.fit_transform_splits(
            train_df, val_df, test_df
        )

        # Kiểm tra is_fitted
        assert pp.is_fitted is True
        assert pp.age_scaler.mean_ is not None

    def test_val_uses_train_scaler(self, train_df, val_df, test_df):
        """Val set phải dùng mean/std của train, không tự tính riêng"""
        pp = TabularPreprocessor()
        pp.fit_transform_splits(train_df, val_df, test_df)

        # mean từ train scaler
        train_mean = pp.age_scaler.mean_[0]

        # Nếu val age = train mean → age_scaled phải ≈ 0
        df_test = make_df(n=5)
        df_test["dx"] = "nv"
        df_test["age"] = train_mean  # Set age = train mean
        df_test["sex"] = "male"
        df_test["localization"] = "back"
        df_test["dx_type"] = "histo"
        clean = pp._clean(df_test)
        encoded = pp._encode_features(clean, fit=False)

        # age_scaled phải ≈ 0 (vì age == mean → (age - mean) / std = 0)
        assert abs(encoded["age_scaled"].mean()) < 0.01

    def test_returns_dataframes(self, train_df, val_df, test_df):
        pp = TabularPreprocessor()
        results = pp.fit_transform_splits(train_df, val_df, test_df)
        assert len(results) == 3
        for df in results:
            assert isinstance(df, pd.DataFrame)

    def test_all_splits_have_same_columns(self, train_df, val_df, test_df):
        pp = TabularPreprocessor()
        train_f, val_f, test_f = pp.fit_transform_splits(train_df, val_df, test_df)
        assert set(train_f.columns) == set(val_f.columns) == set(test_f.columns)


# ══════════════════════════════════════════════════════════
# TEST: get_scaler_params
# ══════════════════════════════════════════════════════════
class TestGetScalerParams:
    def test_raises_if_not_fitted(self):
        pp = TabularPreprocessor()
        with pytest.raises(RuntimeError, match="Chưa fit"):
            pp.get_scaler_params()

    def test_returns_required_keys(self, fitted_preprocessor):
        params = fitted_preprocessor.get_scaler_params()
        assert "age_scaler"     in params
        assert "age_imputer"    in params
        assert "feature_names"  in params
        assert "n_features"     in params
        assert "fitted_at"      in params

    def test_n_features_matches_feature_names(self, fitted_preprocessor):
        params = fitted_preprocessor.get_scaler_params()
        assert params["n_features"] == len(params["feature_names"])

    def test_age_scaler_params_are_valid(self, fitted_preprocessor):
        params = fitted_preprocessor.get_scaler_params()
        scaler = params["age_scaler"]
        assert isinstance(scaler["mean"],  float)
        assert isinstance(scaler["scale"], float)
        assert scaler["scale"] > 0  # std không âm và không bằng 0

    def test_params_are_json_serializable(self, fitted_preprocessor):
        """Params phải serialize được sang JSON để lưu lên MinIO"""
        import json
        params = fitted_preprocessor.get_scaler_params()
        json_str = json.dumps(params)  # Không raise exception
        assert len(json_str) > 0
