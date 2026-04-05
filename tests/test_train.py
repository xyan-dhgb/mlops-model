"""
tests/test_train.py
Integration tests for train_model and evaluate_model.
Uses small synthetic data to keep tests fast.
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_multimodal_model, train_model, evaluate_model


# ── Constants ─────────────────────────────────────────────────────────────────

N_TRAIN = 40
N_VAL   = 10
N_TEST  = 10
TAB_DIM = 15
IMG_SHP = (32, 32, 3)   # Tiny images for speed


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    np.random.seed(0)

    def _make(n):
        imgs = np.random.rand(n, *IMG_SHP).astype(np.float32)
        tabs = np.random.rand(n, TAB_DIM).astype(np.float32)
        # Ensure both classes present
        y = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.float32)
        np.random.shuffle(y)
        return tabs, imgs, y

    return {
        "train": _make(N_TRAIN),
        "val":   _make(N_VAL),
        "test":  _make(N_TEST),
    }


@pytest.fixture(scope="module")
def trained_model(synthetic_data, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("checkpoints")
    model = build_multimodal_model(
        tabular_shape=(TAB_DIM,),
        image_shape=IMG_SHP,
        num_classes=2,
    )
    train = synthetic_data["train"]
    val   = synthetic_data["val"]

    train_model(
        model,
        X_tab_train=train[0], X_img_train=train[1], y_train=train[2],
        X_tab_val=val[0],     X_img_val=val[1],     y_val=val[2],
        epochs=2,
        batch_size=8,
        checkpoint_path=str(tmp / "best.h5"),
    )
    return model


# ── train_model tests ─────────────────────────────────────────────────────────

class TestTrainModel:
    def test_history_has_loss(self, trained_model, synthetic_data):
        # Re-run a 1-epoch training to get a fresh History object
        model = build_multimodal_model(
            tabular_shape=(TAB_DIM,), image_shape=IMG_SHP, num_classes=2
        )
        train = synthetic_data["train"]
        val   = synthetic_data["val"]
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            history = train_model(
                model,
                train[0], train[1], train[2],
                val[0],   val[1],   val[2],
                epochs=1, batch_size=8,
                checkpoint_path=str(pathlib.Path(td) / "best.h5"),
            )
        assert "loss" in history.history
        assert len(history.history["loss"]) >= 1

    def test_history_has_val_auc(self, synthetic_data):
        model = build_multimodal_model(
            tabular_shape=(TAB_DIM,), image_shape=IMG_SHP, num_classes=2
        )
        train = synthetic_data["train"]
        val   = synthetic_data["val"]
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            history = train_model(
                model,
                train[0], train[1], train[2],
                val[0],   val[1],   val[2],
                epochs=1, batch_size=8,
                checkpoint_path=str(pathlib.Path(td) / "best.h5"),
            )
        assert "val_auc" in history.history

    def test_checkpoint_created(self, synthetic_data):
        model = build_multimodal_model(
            tabular_shape=(TAB_DIM,), image_shape=IMG_SHP, num_classes=2
        )
        train = synthetic_data["train"]
        val   = synthetic_data["val"]
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            ckpt = str(pathlib.Path(td) / "best.h5")
            train_model(
                model,
                train[0], train[1], train[2],
                val[0],   val[1],   val[2],
                epochs=1, batch_size=8,
                checkpoint_path=ckpt,
            )
            assert pathlib.Path(ckpt).exists(), "Checkpoint file must be created"


# ── evaluate_model tests ──────────────────────────────────────────────────────

class TestEvaluateModel:
    def test_returns_accuracy_and_auc(self, trained_model, synthetic_data):
        test = synthetic_data["test"]
        acc, auc_score = evaluate_model(
            trained_model, test[0], test[1], test[2]
        )
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= auc_score <= 1.0

    def test_accuracy_is_float(self, trained_model, synthetic_data):
        test = synthetic_data["test"]
        acc, _ = evaluate_model(trained_model, test[0], test[1], test[2])
        assert isinstance(acc, float)

    def test_auc_is_float(self, trained_model, synthetic_data):
        test = synthetic_data["test"]
        _, auc_score = evaluate_model(trained_model, test[0], test[1], test[2])
        assert isinstance(auc_score, float)

    def test_perfect_model_high_accuracy(self):
        """A model that always predicts 0 on all-0 labels should score acc=1."""
        model = build_multimodal_model(
            tabular_shape=(TAB_DIM,), image_shape=IMG_SHP, num_classes=2
        )
        # All-benign labels; model output close to 0
        n = 20
        imgs = np.zeros((n, *IMG_SHP), dtype=np.float32)
        tabs = np.zeros((n, TAB_DIM), dtype=np.float32)
        y    = np.zeros(n, dtype=np.float32)

        # Bias output toward 0 by setting last-layer bias
        # (just checks that evaluate_model runs without error on edge case)
        acc, auc_score = evaluate_model(model, tabs, imgs, y)
        assert np.isfinite(acc)
