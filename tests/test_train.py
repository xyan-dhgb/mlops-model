"""
Unit tests for train.py
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.model import build_multimodal_model
from src.train import evaluate_model, predict, train_model


# ─────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────
TAB_SHAPE = (4,)
IMG_SHAPE = (16, 16, 3)  # tiny images to keep tests fast
N_CLASSES = 3
BATCH = 8


def _make_batch(n: int) -> tuple:
    np.random.seed(7)
    X_tab = np.random.randn(n, *TAB_SHAPE).astype(np.float32)
    X_img = np.random.randn(n, *IMG_SHAPE).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, size=n)
    return X_tab, X_img, y


def _tiny_model():
    return build_multimodal_model(TAB_SHAPE, IMG_SHAPE, N_CLASSES)


# ─────────────────────────────────────────────
# Tests: train_model
# ─────────────────────────────────────────────
class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model()
        self.X_tab_tr, self.X_img_tr, self.y_tr = _make_batch(16)
        self.X_tab_val, self.X_img_val, self.y_val = _make_batch(8)

    def test_returns_history(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmp:
            history = train_model(
                self.model,
                self.X_tab_tr, self.X_img_tr, self.y_tr,
                self.X_tab_val, self.X_img_val, self.y_val,
                epochs=1,
                batch_size=4,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                log_dir=os.path.join(tmp, "logs"),
            )
            self.assertIsInstance(history, tf.keras.callbacks.History)

    def test_history_has_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = train_model(
                self.model,
                self.X_tab_tr, self.X_img_tr, self.y_tr,
                self.X_tab_val, self.X_img_val, self.y_val,
                epochs=1,
                batch_size=4,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                log_dir=os.path.join(tmp, "logs"),
            )
            self.assertIn("accuracy", history.history)
            self.assertIn("val_accuracy", history.history)

    def test_checkpoint_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = os.path.join(tmp, "ckpt")
            train_model(
                self.model,
                self.X_tab_tr, self.X_img_tr, self.y_tr,
                self.X_tab_val, self.X_img_val, self.y_val,
                epochs=1,
                batch_size=4,
                checkpoint_dir=ckpt_dir,
                log_dir=os.path.join(tmp, "logs"),
            )
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "best_model.h5")))


# ─────────────────────────────────────────────
# Tests: evaluate_model
# ─────────────────────────────────────────────
class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model()
        self.X_tab, self.X_img, self.y = _make_batch(8)

    def test_returns_dict(self):
        metrics = evaluate_model(self.model, self.X_tab, self.X_img, self.y)
        self.assertIsInstance(metrics, dict)

    def test_has_loss_and_accuracy(self):
        metrics = evaluate_model(self.model, self.X_tab, self.X_img, self.y)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)

    def test_accuracy_range(self):
        metrics = evaluate_model(self.model, self.X_tab, self.X_img, self.y)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_loss_positive(self):
        metrics = evaluate_model(self.model, self.X_tab, self.X_img, self.y)
        self.assertGreater(metrics["loss"], 0.0)


# ─────────────────────────────────────────────
# Tests: predict
# ─────────────────────────────────────────────
class TestPredict(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model()
        self.X_tab, self.X_img, _ = _make_batch(4)

    def test_output_shape(self):
        preds = predict(self.model, self.X_tab, self.X_img)
        self.assertEqual(preds.shape, (4, N_CLASSES))

    def test_probabilities_sum_to_one(self):
        preds = predict(self.model, self.X_tab, self.X_img)
        np.testing.assert_array_almost_equal(
            preds.sum(axis=1), np.ones(4), decimal=5
        )

    def test_no_nan_in_output(self):
        preds = predict(self.model, self.X_tab, self.X_img)
        self.assertFalse(np.isnan(preds).any())


if __name__ == "__main__":
    unittest.main()
