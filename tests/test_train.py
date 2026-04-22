"""
tests/test_train.py
====================
Integration smoke-test for the full training pipeline.
Runs with tiny synthetic data – no real HDF5/CSV required.
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (
    build_tabular_features,
    oversample_malignant,
    stratified_split,
)
from model import (
    IMAGE_SHAPE,
    build_multimodal_model,
    evaluate_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 50   # tiny dataset for fast CI
N_BEN, N_MAL = 45, 5


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Return (X_tab, X_img, y) with correct shapes."""
    rng = np.random.default_rng(42)
    n = N_BEN + N_MAL
    X_img = rng.random((n, 224, 224, 3), dtype=np.float64).astype(np.float32)
    X_tab = rng.random((n, 15)).astype(np.float32)
    y = np.array([0] * N_BEN + [1] * N_MAL, dtype=np.int32)
    return X_tab, X_img, y


@pytest.fixture(scope="module")
def compiled_model():
    model, backbone = build_multimodal_model(
        tabular_shape=(15,),
        image_shape=IMAGE_SHAPE,
        freeze_backbone=True,
    )
    return model, backbone


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestTrainingPipeline:
    def test_stratified_split_sizes(self, synthetic_dataset):
        X_tab, X_img, y = synthetic_dataset
        splits = stratified_split(X_tab, X_img, y)
        total = sum(len(splits[k][2]) for k in ["train", "val", "test"])
        assert total == len(y)

    def test_oversampling_increases_size(self, synthetic_dataset):
        X_tab, X_img, y = synthetic_dataset
        X_img_os, X_tab_os, y_os = oversample_malignant(
            X_img, X_tab, y, target_ratio=0.25, strong_aug=False
        )
        assert len(y_os) >= len(y)

    def test_model_output_shape(self, compiled_model, synthetic_dataset):
        model, _ = compiled_model
        X_tab, X_img, y = synthetic_dataset
        preds = model.predict(
            {"image_input": X_img[:4], "tabular_input": X_tab[:4]},
            verbose=0,
        )
        assert preds.shape == (4, 1)

    def test_phase1_training_runs(self, compiled_model, synthetic_dataset):
        """Phase 1 training should complete without errors."""
        model, backbone = compiled_model
        X_tab, X_img, y = synthetic_dataset

        splits = stratified_split(X_tab, X_img, y)
        X_tab_tr, X_img_tr, y_tr = splits["train"]
        X_tab_vl, X_img_vl, y_vl = splits["val"]

        with tempfile.TemporaryDirectory() as tmpdir:
            h1, h2 = train_model(
                model, backbone,
                X_tab_tr, X_img_tr, y_tr,
                X_tab_vl, X_img_vl, y_vl,
                phase1_epochs=1,
                phase2_epochs=1,
                batch_size=8,
                run_phase2=False,   # skip phase2 for speed
                checkpoint_dir=tmpdir,
                oversample_ratio=0.25,
            )
        assert "val_auc" in h1.history
        assert h2 is None  # phase2 skipped

    def test_evaluate_model_keys(self, compiled_model, synthetic_dataset):
        model, _ = compiled_model
        X_tab, X_img, y = synthetic_dataset
        splits = stratified_split(X_tab, X_img, y)
        X_tab_te, X_img_te, y_te = splits["test"]

        metrics = evaluate_model(
            model, X_tab_te, X_img_te, y_te,
            auto_tune_threshold=True,
        )
        required = {"auc_roc", "pauc_tpr80", "f1_malignant", "recall_malignant", "threshold"}
        assert required.issubset(metrics.keys())

    def test_all_metrics_in_valid_range(self, compiled_model, synthetic_dataset):
        model, _ = compiled_model
        X_tab, X_img, y = synthetic_dataset
        splits = stratified_split(X_tab, X_img, y)
        X_tab_te, X_img_te, y_te = splits["test"]

        metrics = evaluate_model(model, X_tab_te, X_img_te, y_te)
        for key in ["auc_roc", "pauc_tpr80", "f1_malignant", "recall_malignant"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"
