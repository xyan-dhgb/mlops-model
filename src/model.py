"""
src/model.py
Public API for model construction and XAI.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Multimodal.models.multimodal_model import build_multimodal_model
from Multimodal.training.train import (
    train_model,
    evaluate_model,
    plot_training_history,
    GradCAMExplainer,
    SHAPMetadataExplainer,
    MultimodalXAIRunner,
)

__all__ = [
    "build_multimodal_model",
    "train_model",
    "evaluate_model",
    "plot_training_history",
    "GradCAMExplainer",
    "SHAPMetadataExplainer",
    "MultimodalXAIRunner",
]
