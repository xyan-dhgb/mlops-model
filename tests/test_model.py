"""
tests/test_model.py
Unit tests for model architecture and forward pass.
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_multimodal_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

TABULAR_DIM = 30
IMAGE_SHAPE = (64, 64, 3)  # Small size for fast tests
BATCH = 4


@pytest.fixture(scope="module")
def binary_model():
    return build_multimodal_model(
        tabular_shape=(TABULAR_DIM,),
        image_shape=IMAGE_SHAPE,
        num_classes=2,
    )


@pytest.fixture(scope="module")
def multiclass_model():
    return build_multimodal_model(
        tabular_shape=(TABULAR_DIM,),
        image_shape=IMAGE_SHAPE,
        num_classes=4,
    )


@pytest.fixture()
def batch_inputs():
    np.random.seed(7)
    return (
        np.random.rand(BATCH, *IMAGE_SHAPE).astype(np.float32),
        np.random.rand(BATCH, TABULAR_DIM).astype(np.float32),
    )


# ── Architecture tests ────────────────────────────────────────────────────────

class TestModelArchitecture:
    def test_model_has_two_inputs(self, binary_model):
        assert len(binary_model.inputs) == 2

    def test_input_names(self, binary_model):
        names = {inp.name.split(":")[0] for inp in binary_model.inputs}
        assert "image_input" in names
        assert "tabular_input" in names

    def test_binary_output_shape(self, binary_model):
        assert binary_model.output_shape == (None, 1)

    def test_multiclass_output_shape(self, multiclass_model):
        assert multiclass_model.output_shape == (None, 4)

    def test_model_is_compiled(self, binary_model):
        assert binary_model.optimizer is not None

    def test_model_has_auc_metric(self, binary_model):
        metric_names = [m.name for m in binary_model.metrics]
        assert "auc" in metric_names


# ── Forward pass tests ────────────────────────────────────────────────────────

class TestForwardPass:
    def test_binary_output_range(self, binary_model, batch_inputs):
        imgs, tabs = batch_inputs
        preds = binary_model.predict(
            {"image_input": imgs, "tabular_input": tabs}, verbose=0
        )
        assert preds.shape == (BATCH, 1)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_multiclass_output_sums_to_one(self, multiclass_model, batch_inputs):
        imgs, tabs = batch_inputs
        preds = multiclass_model.predict(
            {"image_input": imgs, "tabular_input": tabs}, verbose=0
        )
        assert preds.shape == (BATCH, 4)
        np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)

    def test_batch_size_one(self, binary_model):
        img = np.random.rand(1, *IMAGE_SHAPE).astype(np.float32)
        tab = np.random.rand(1, TABULAR_DIM).astype(np.float32)
        pred = binary_model.predict(
            {"image_input": img, "tabular_input": tab}, verbose=0
        )
        assert pred.shape == (1, 1)


# ── Training step test ────────────────────────────────────────────────────────

class TestTrainingStep:
    def test_single_training_step(self, binary_model, batch_inputs):
        imgs, tabs = batch_inputs
        y = np.random.randint(0, 2, BATCH).astype(np.float32)
        result = binary_model.train_on_batch(
            {"image_input": imgs, "tabular_input": tabs}, y
        )
        loss = result[0] if isinstance(result, (list, tuple)) else result
        assert np.isfinite(loss), "Loss should be finite after one training step"

    def test_weights_change_after_training(self, batch_inputs):
        model = build_multimodal_model(
            tabular_shape=(TABULAR_DIM,),
            image_shape=IMAGE_SHAPE,
            num_classes=2,
        )
        imgs, tabs = batch_inputs
        y = np.random.randint(0, 2, BATCH).astype(np.float32)

        # Snapshot weights before
        w_before = [w.numpy().copy() for w in model.trainable_weights[:3]]
        model.train_on_batch(
            {"image_input": imgs, "tabular_input": tabs}, y
        )
        w_after = [w.numpy() for w in model.trainable_weights[:3]]

        any_changed = any(
            not np.allclose(b, a) for b, a in zip(w_before, w_after)
        )
        assert any_changed, "Weights should change after a training step"
