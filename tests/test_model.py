"""
Unit tests for model.py
Tests verify model architecture, compilation, and forward pass shapes
without requiring GPU or real data.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress TF logs during testing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.model import build_multimodal_model, build_image_branch, build_tabular_branch, get_model_summary


# ─────────────────────────────────────────────
# Tests: build_image_branch
# ─────────────────────────────────────────────
class TestImageBranch(unittest.TestCase):
    def test_output_is_tuple(self):
        result = build_image_branch((32, 32, 3))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_input_shape_recorded(self):
        inp, _ = build_image_branch((64, 64, 3))
        self.assertEqual(tuple(inp.shape[1:]), (64, 64, 3))


# ─────────────────────────────────────────────
# Tests: build_tabular_branch
# ─────────────────────────────────────────────
class TestTabularBranch(unittest.TestCase):
    def test_output_is_tuple(self):
        result = build_tabular_branch((4,))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_input_shape_recorded(self):
        inp, _ = build_tabular_branch((8,))
        self.assertEqual(tuple(inp.shape[1:]), (8,))


# ─────────────────────────────────────────────
# Tests: build_multimodal_model
# ─────────────────────────────────────────────
class TestBuildMultimodalModel(unittest.TestCase):
    def setUp(self):
        self.tabular_shape = (4,)
        self.image_shape = (32, 32, 3)
        self.num_classes = 3

    def test_model_compiles(self):
        model = build_multimodal_model(
            self.tabular_shape, self.image_shape, self.num_classes
        )
        self.assertIsNotNone(model)

    def test_output_shape(self):
        model = build_multimodal_model(
            self.tabular_shape, self.image_shape, self.num_classes
        )
        self.assertEqual(model.output_shape, (None, self.num_classes))

    def test_model_has_two_inputs(self):
        model = build_multimodal_model(
            self.tabular_shape, self.image_shape, self.num_classes
        )
        self.assertEqual(len(model.inputs), 2)

    def test_input_names(self):
        model = build_multimodal_model(
            self.tabular_shape, self.image_shape, self.num_classes
        )
        input_names = [inp.name for inp in model.inputs]
        self.assertTrue(any("image" in n for n in input_names))
        self.assertTrue(any("tabular" in n for n in input_names))

    def test_forward_pass_shape(self):
        model = build_multimodal_model(
            self.tabular_shape, self.image_shape, self.num_classes
        )
        batch_size = 4
        dummy_img = np.random.randn(batch_size, *self.image_shape).astype(np.float32)
        dummy_tab = np.random.randn(batch_size, *self.tabular_shape).astype(np.float32)
        preds = model.predict(
            {"image_input": dummy_img, "tabular_input": dummy_tab}, verbose=0
        )
        self.assertEqual(preds.shape, (batch_size, self.num_classes))

    def test_probabilities_sum_to_one(self):
        model = build_multimodal_model(
            self.tabular_shape, self.image_shape, self.num_classes
        )
        dummy_img = np.random.randn(2, *self.image_shape).astype(np.float32)
        dummy_tab = np.random.randn(2, *self.tabular_shape).astype(np.float32)
        preds = model.predict(
            {"image_input": dummy_img, "tabular_input": dummy_tab}, verbose=0
        )
        np.testing.assert_array_almost_equal(
            preds.sum(axis=1), np.ones(2), decimal=5
        )

    def test_num_classes_binary(self):
        model = build_multimodal_model(self.tabular_shape, self.image_shape, 2)
        self.assertEqual(model.output_shape, (None, 2))

    def test_different_tabular_dim(self):
        model = build_multimodal_model((10,), self.image_shape, self.num_classes)
        dummy_img = np.random.randn(2, *self.image_shape).astype(np.float32)
        dummy_tab = np.random.randn(2, 10).astype(np.float32)
        preds = model.predict(
            {"image_input": dummy_img, "tabular_input": dummy_tab}, verbose=0
        )
        self.assertEqual(preds.shape[1], self.num_classes)


# ─────────────────────────────────────────────
# Tests: get_model_summary
# ─────────────────────────────────────────────
class TestGetModelSummary(unittest.TestCase):
    def test_returns_string(self):
        model = build_multimodal_model((4,), (32, 32, 3), 3)
        summary = get_model_summary(model)
        self.assertIsInstance(summary, str)

    def test_summary_contains_model_name(self):
        model = build_multimodal_model((4,), (32, 32, 3), 3)
        summary = get_model_summary(model)
        self.assertIn("SkinCancerMultimodal", summary)


if __name__ == "__main__":
    unittest.main()
