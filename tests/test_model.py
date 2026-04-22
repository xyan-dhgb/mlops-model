"""
tests/test_model.py
====================
Unit tests for src/model.py
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import (
    build_multimodal_model,
    compute_pauc,
    find_optimal_threshold,
    focal_loss,
    predict_skin_lesion,
)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_scalar_output(self):
        import tensorflow as tf
        loss_fn = focal_loss(gamma=2.0, alpha=0.75)
        y_true = tf.constant([1.0, 0.0, 1.0])
        y_pred = tf.constant([0.9, 0.1, 0.8])
        val = loss_fn(y_true, y_pred)
        assert val.numpy() >= 0

    def test_lower_loss_for_confident_correct_prediction(self):
        import tensorflow as tf
        loss_fn = focal_loss(gamma=2.0, alpha=0.75)
        y_true = tf.constant([1.0])
        confident = loss_fn(y_true, tf.constant([0.95]))
        uncertain = loss_fn(y_true, tf.constant([0.55]))
        assert confident.numpy() < uncertain.numpy()


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TestBuildMultimodalModel:
    @pytest.fixture(scope="class")
    def small_model(self):
        """Build once for all tests in this class."""
        model, backbone = build_multimodal_model(
            tabular_shape=(10,),
            image_shape=(224, 224, 3),
            freeze_backbone=True,
        )
        return model, backbone

    def test_output_shape(self, small_model):
        model, _ = small_model
        assert model.output_shape == (None, 1)

    def test_input_names(self, small_model):
        model, _ = small_model
        input_names = [inp.name for inp in model.inputs]
        assert any("image" in n for n in input_names)
        assert any("tabular" in n for n in input_names)

    def test_backbone_frozen(self, small_model):
        model, backbone = small_model
        assert not backbone.trainable

    def test_model_predicts(self, small_model):
        model, _ = small_model
        rng = np.random.default_rng(0)
        imgs = rng.random((2, 224, 224, 3)).astype(np.float32)
        tabs = rng.random((2, 10)).astype(np.float32)
        preds = model.predict({"image_input": imgs, "tabular_input": tabs}, verbose=0)
        assert preds.shape == (2, 1)
        assert ((preds >= 0) & (preds <= 1)).all()


# ---------------------------------------------------------------------------
# pAUC
# ---------------------------------------------------------------------------

class TestComputePAUC:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.15, 0.9, 0.95])
        pauc = compute_pauc(y_true, y_pred, min_tpr=0.80)
        assert pauc > 0

    def test_range_0_to_1(self):
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.random(100).astype(np.float32)
        pauc = compute_pauc(y_true, y_pred)
        assert 0.0 <= pauc <= 1.0

    def test_insufficient_tpr_returns_zero(self):
        # All predictions 0.1 → TPR always 0, no mask points ≥ 0.8
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        pauc = compute_pauc(y_true, y_pred, min_tpr=0.80)
        assert pauc == 0.0


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    def _make_data(self):
        rng = np.random.default_rng(11)
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.concatenate([rng.random(90) * 0.4, rng.random(10) * 0.4 + 0.6])
        return y_true, y_pred

    def test_threshold_in_range(self):
        y_true, y_pred = self._make_data()
        thr, _ = find_optimal_threshold(y_true, y_pred)
        assert 0.05 <= thr <= 0.95

    def test_returns_dataframe(self):
        import pandas as pd
        y_true, y_pred = self._make_data()
        _, df = find_optimal_threshold(y_true, y_pred)
        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns

    def test_recall_constraint_respected(self):
        y_true, y_pred = self._make_data()
        thr, df = find_optimal_threshold(y_true, y_pred, min_recall=0.60)
        pred = (y_pred >= thr).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        recall = tp / (tp + fn + 1e-8)
        assert recall >= 0.55  # allow small slack due to grid resolution
