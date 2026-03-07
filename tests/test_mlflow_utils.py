"""
tests/unit/test_mlflow_utils.py
Unit tests cho MLflow utilities: EpochLogger, ModelRegistry, helper functions
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.preprocessing.mlflow_utils import (
    EpochLogger,
    ModelRegistry,
    log_class_metrics,
)


# ══════════════════════════════════════════════════════════
# TEST: EpochLogger
# ══════════════════════════════════════════════════════════
class TestEpochLogger:
    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_log_epoch_calls_mlflow_log_metric(self, mock_mlflow):
        logger = EpochLogger()
        metrics = {"train/loss": 0.5, "val/accuracy": 0.85}
        logger.log_epoch(epoch=1, metrics=metrics)

        # Kiểm tra mlflow.log_metric được gọi đúng số lần
        assert mock_mlflow.log_metric.call_count == 2

    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_log_epoch_passes_correct_step(self, mock_mlflow):
        logger = EpochLogger()
        logger.log_epoch(epoch=5, metrics={"train/loss": 0.3})

        call_args = mock_mlflow.log_metric.call_args
        # step argument phải là 5
        assert call_args[1]["step"] == 5 or call_args[0][-1] == 5

    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_history_updated(self, mock_mlflow):
        logger = EpochLogger()
        logger.log_epoch(0, {"loss": 0.9})
        logger.log_epoch(1, {"loss": 0.7})
        logger.log_epoch(2, {"loss": 0.5})

        assert len(logger.history) == 3
        assert logger.history[0]["epoch"] == 0
        assert logger.history[2]["loss"]  == 0.5

    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_log_best_logs_with_prefix(self, mock_mlflow):
        logger = EpochLogger()
        logger.log_best(
            best_epoch=10,
            best_metrics={"val/f1_macro": 0.82, "val/accuracy": 0.88}
        )

        calls = [str(c) for c in mock_mlflow.log_metric.call_args_list]
        # Kiểm tra prefix "best/" xuất hiện
        assert any("best/" in c for c in calls)

    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_log_best_epoch_logged(self, mock_mlflow):
        logger = EpochLogger()
        logger.log_best(best_epoch=7, best_metrics={"acc": 0.9})
        calls = mock_mlflow.log_metric.call_args_list
        metric_names = [c[0][0] for c in calls]
        assert "best_epoch" in metric_names


# ══════════════════════════════════════════════════════════
# TEST: log_class_metrics
# ══════════════════════════════════════════════════════════
class TestLogClassMetrics:
    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_logs_all_6_classes(self, mock_mlflow):
        report = {
            "ACK": {"precision": 0.80, "recall": 0.75, "f1-score": 0.77},
            "BCC": {"precision": 0.90, "recall": 0.85, "f1-score": 0.87},
            "MEL": {"precision": 0.70, "recall": 0.65, "f1-score": 0.67},
            "NEV": {"precision": 0.95, "recall": 0.93, "f1-score": 0.94},
            "SCC": {"precision": 0.60, "recall": 0.55, "f1-score": 0.57},
            "SEK": {"precision": 0.85, "recall": 0.80, "f1-score": 0.82},
            "macro avg": {"precision": 0.80, "recall": 0.76, "f1-score": 0.78},
            "weighted avg": {"precision": 0.82, "recall": 0.78, "f1-score": 0.80},
        }
        log_class_metrics(report, prefix="val/")

        # 6 classes × 3 metrics = 18 + macro + weighted = 18+6 = 24
        assert mock_mlflow.log_metric.call_count >= 18

    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_prefix_applied_correctly(self, mock_mlflow):
        report = {
            "MEL": {"precision": 0.80, "recall": 0.75, "f1-score": 0.77},
        }
        log_class_metrics(report, prefix="test/")

        call_names = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        assert all(n.startswith("test/") for n in call_names if "MEL" in n)

    @patch("src.preprocessing.mlflow_utils.mlflow")
    def test_missing_class_handled_gracefully(self, mock_mlflow):
        """Report thiếu 1 class → không crash"""
        report = {
            "MEL": {"precision": 0.80, "recall": 0.75, "f1-score": 0.77},
            # ACK, BCC... bị thiếu
        }
        # Không raise exception
        log_class_metrics(report, prefix="val/")


# ══════════════════════════════════════════════════════════
# TEST: ModelRegistry
# ══════════════════════════════════════════════════════════
class TestModelRegistry:
    @patch("src.preprocessing.mlflow_utils.mlflow")
    @patch("src.preprocessing.mlflow_utils.MlflowClient")
    def test_register_calls_mlflow_register(self, MockClient, mock_mlflow):
        mock_mlflow.register_model.return_value = MagicMock(version="3")
        MockClient.return_value = MagicMock()

        registry = ModelRegistry()
        version = registry.register(
            run_id="abc123",
            model_name="cancer-multimodal",
            artifact_path="model",
            description="Test v1"
        )

        mock_mlflow.register_model.assert_called_once_with(
            model_uri="runs:/abc123/model",
            name="cancer-multimodal"
        )
        assert version == "3"

    @patch("src.preprocessing.mlflow_utils.mlflow")
    @patch("src.preprocessing.mlflow_utils.MlflowClient")
    def test_promote_to_staging(self, MockClient, mock_mlflow):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        registry = ModelRegistry()
        registry.promote_to_staging("cancer-multimodal", "3")

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="cancer-multimodal",
            version="3",
            stage="Staging",
            archive_existing_versions=False
        )

    @patch("src.preprocessing.mlflow_utils.mlflow")
    @patch("src.preprocessing.mlflow_utils.MlflowClient")
    def test_promote_to_production_archives_old(self, MockClient, mock_mlflow):
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        registry = ModelRegistry()
        registry.promote_to_production("cancer-multimodal", "3", archive_old=True)

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="cancer-multimodal",
            version="3",
            stage="Production",
            archive_existing_versions=True
        )
