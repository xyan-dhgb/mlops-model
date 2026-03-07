"""
Unit tests for data_preprocessing.py
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image


# ── Patch heavy imports so tests run without GPU ─────────────────────────────
import sys

# Make sure src is on path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import (
    encode_tabular_features,
    load_csv_data,
    load_image,
    preprocess_csv_data,
    split_dataset,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def make_sample_df(n: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        {
            "img_id": [f"img_{i}.png" for i in range(n)],
            "diagnostic": np.random.choice(["BCC", "MEL", "SCC"], size=n),
            "age": np.random.randint(20, 80, size=n).astype(float),
            "fitspatrick": np.random.randint(1, 6, size=n).astype(float),
            "diameter_1": np.random.uniform(2, 20, size=n),
            "diameter_2": np.random.uniform(2, 20, size=n),
        }
    )


# ─────────────────────────────────────────────
# Tests: load_csv_data
# ─────────────────────────────────────────────
class TestLoadCSVData(unittest.TestCase):
    def test_load_valid_csv(self):
        df = make_sample_df()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = f.name
        try:
            loaded = load_csv_data(tmp_path)
            self.assertEqual(loaded.shape, df.shape)
        finally:
            os.unlink(tmp_path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_csv_data("/nonexistent/path/metadata.csv")

    def test_returns_dataframe(self):
        df = make_sample_df(5)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = f.name
        try:
            result = load_csv_data(tmp_path)
            self.assertIsInstance(result, pd.DataFrame)
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Tests: preprocess_csv_data
# ─────────────────────────────────────────────
class TestPreprocessCSVData(unittest.TestCase):
    def setUp(self):
        self.df = make_sample_df(30)

    def test_returns_tuple(self):
        result = preprocess_csv_data(self.df)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_no_missing_after_preprocessing(self):
        df_with_nan = self.df.copy()
        df_with_nan.loc[0, "age"] = np.nan
        df_with_nan.loc[1, "fitspatrick"] = np.nan
        processed, report = preprocess_csv_data(df_with_nan)
        self.assertEqual(report["missing_after"], 0)

    def test_report_contains_required_keys(self):
        _, report = preprocess_csv_data(self.df)
        for key in ("initial_shape", "final_shape", "missing_before", "missing_after"):
            self.assertIn(key, report)

    def test_column_names_normalised(self):
        df_ugly = self.df.rename(columns={"age": "  Age  ", "fitspatrick": "FitSpatrick"})
        processed, _ = preprocess_csv_data(df_ugly)
        self.assertIn("age", processed.columns)
        self.assertIn("fitspatrick", processed.columns)

    def test_shape_preserved(self):
        processed, _ = preprocess_csv_data(self.df)
        self.assertEqual(processed.shape[0], self.df.shape[0])


# ─────────────────────────────────────────────
# Tests: encode_tabular_features
# ─────────────────────────────────────────────
class TestEncodeTabularFeatures(unittest.TestCase):
    def setUp(self):
        self.df, _ = preprocess_csv_data(make_sample_df(30))

    def test_output_shapes(self):
        X, y, scaler, le = encode_tabular_features(self.df, fit=True)
        self.assertEqual(X.shape[0], self.df.shape[0])
        self.assertEqual(len(y), self.df.shape[0])

    def test_labels_integer(self):
        _, y, _, _ = encode_tabular_features(self.df, fit=True)
        self.assertTrue(np.issubdtype(y.dtype, np.integer))

    def test_features_float32(self):
        X, _, _, _ = encode_tabular_features(self.df, fit=True)
        self.assertEqual(X.dtype, np.float32)

    def test_transform_mode_consistent(self):
        X_fit, y_fit, scaler, le = encode_tabular_features(self.df, fit=True)
        X_tf, y_tf, _, _ = encode_tabular_features(
            self.df, fit=False, scaler=scaler, label_encoder=le
        )
        np.testing.assert_array_almost_equal(X_fit, X_tf)


# ─────────────────────────────────────────────
# Tests: load_image
# ─────────────────────────────────────────────
class TestLoadImage(unittest.TestCase):
    def _make_png(self, size=(50, 50)) -> str:
        img = Image.fromarray(
            np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            return f.name

    def test_output_shape(self):
        path = self._make_png()
        try:
            arr = load_image(path, target_size=(64, 64))
            self.assertEqual(arr.shape, (64, 64, 3))
        finally:
            os.unlink(path)

    def test_pixel_range(self):
        path = self._make_png()
        try:
            arr = load_image(path)
            self.assertGreaterEqual(arr.min(), 0.0)
            self.assertLessEqual(arr.max(), 1.0)
        finally:
            os.unlink(path)

    def test_missing_image_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_image("/no/such/image.png")

    def test_float32_dtype(self):
        path = self._make_png()
        try:
            arr = load_image(path)
            self.assertEqual(arr.dtype, np.float32)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────
# Tests: split_dataset
# ─────────────────────────────────────────────
class TestSplitDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 100
        self.X_tab = np.random.randn(n, 4).astype(np.float32)
        self.X_img = np.random.randn(n, 32, 32, 3).astype(np.float32)
        self.y = np.random.randint(0, 3, size=n)

    def test_split_keys_exist(self):
        splits = split_dataset(self.X_tab, self.X_img, self.y)
        expected_keys = {
            "X_tab_train", "X_tab_val", "X_tab_test",
            "X_img_train", "X_img_val", "X_img_test",
            "y_train", "y_val", "y_test",
        }
        self.assertEqual(set(splits.keys()), expected_keys)

    def test_no_data_leakage(self):
        splits = split_dataset(self.X_tab, self.X_img, self.y)
        total = (
            len(splits["y_train"])
            + len(splits["y_val"])
            + len(splits["y_test"])
        )
        self.assertEqual(total, len(self.y))

    def test_test_size_approx(self):
        splits = split_dataset(self.X_tab, self.X_img, self.y, test_size=0.2)
        self.assertAlmostEqual(len(splits["y_test"]) / len(self.y), 0.2, delta=0.05)

    def test_shapes_consistent(self):
        splits = split_dataset(self.X_tab, self.X_img, self.y)
        self.assertEqual(splits["X_tab_train"].shape[0], splits["X_img_train"].shape[0])
        self.assertEqual(splits["X_tab_train"].shape[0], len(splits["y_train"]))


if __name__ == "__main__":
    unittest.main()
