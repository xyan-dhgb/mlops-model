"""
src/model.py
Model registry helpers: load from MLflow, save checkpoints, adversarial eval.
Used by training scripts and serving layer.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import mlflow.pytorch

from Multimodal.models.multimodal_model import MultimodalSkinClassifier

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
def load_model_from_checkpoint(
    checkpoint_path: str,
    num_classes: int = 7,
    metadata_input_dim: int = 5,
    device: str = "cpu",
) -> MultimodalSkinClassifier:
    """Load model weights from a local .pt checkpoint."""
    model = MultimodalSkinClassifier(
        num_classes=num_classes,
        metadata_input_dim=metadata_input_dim,
        pretrained=False,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    log.info("Loaded checkpoint from %s", checkpoint_path)
    return model


def load_model_from_mlflow(
    model_uri: str,
    device: str = "cpu",
) -> MultimodalSkinClassifier:
    """
    Load registered model from MLflow Registry.
    model_uri examples:
      - "models:/multimodal_skin_cancer_v1/Production"
      - "runs:/<run_id>/model"
    """
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.to(device).eval()
    log.info("Loaded model from MLflow: %s", model_uri)
    return model


# ─────────────────────────────────────────────
# Adversarial Robustness Evaluation
# SecMLOps: FGSM + PGD before production deploy
# ─────────────────────────────────────────────
def fgsm_attack(
    image: torch.Tensor,
    epsilon: float,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """Fast Gradient Sign Method perturbation."""
    return torch.clamp(image + epsilon * gradient.sign(), 0, 1)


@torch.enable_grad()
def evaluate_adversarial_robustness(
    model: MultimodalSkinClassifier,
    images: torch.Tensor,
    metadata: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    pgd_steps: int = 10,
    pgd_alpha: float = 0.007,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model robustness against FGSM and PGD attacks.
    Results should be logged to MLflow before production registration.

    Args:
        epsilon   : max L-inf perturbation (0.03 = ~8/255 for normalized images)
        pgd_steps : PGD iterations
        pgd_alpha : PGD step size

    Returns dict with:
        clean_acc, fgsm_acc, pgd_acc, robustness_gap
    """
    model.eval()
    dev    = torch.device(device)
    images = images.to(dev).float()
    meta   = metadata.to(dev).float()
    labels = labels.to(dev)

    criterion = torch.nn.CrossEntropyLoss()

    # Clean accuracy
    with torch.no_grad():
        clean_logits = model(images, meta)
        clean_acc = (clean_logits.argmax(1) == labels).float().mean().item()

    # FGSM
    imgs_adv = images.clone().requires_grad_(True)
    loss = criterion(model(imgs_adv, meta), labels)
    loss.backward()
    fgsm_imgs = fgsm_attack(images, epsilon, imgs_adv.grad)
    with torch.no_grad():
        fgsm_acc = (model(fgsm_imgs, meta).argmax(1) == labels).float().mean().item()

    # PGD
    pgd_imgs = images.clone() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    pgd_imgs = torch.clamp(pgd_imgs, 0, 1)
    for _ in range(pgd_steps):
        pgd_imgs.requires_grad_(True)
        loss = criterion(model(pgd_imgs, meta), labels)
        loss.backward()
        with torch.no_grad():
            pgd_imgs = pgd_imgs + pgd_alpha * pgd_imgs.grad.sign()
            pgd_imgs = torch.clamp(pgd_imgs, images - epsilon, images + epsilon)
            pgd_imgs = torch.clamp(pgd_imgs, 0, 1)
    with torch.no_grad():
        pgd_acc = (model(pgd_imgs, meta).argmax(1) == labels).float().mean().item()

    results = {
        "clean_accuracy":    round(clean_acc, 4),
        "fgsm_accuracy":     round(fgsm_acc,  4),
        "pgd_accuracy":      round(pgd_acc,   4),
        "robustness_gap":    round(clean_acc - pgd_acc, 4),
        "epsilon":           epsilon,
        "pgd_steps":         pgd_steps,
    }
    log.info("Adversarial eval: %s", results)
    return results


# ─────────────────────────────────────────────
# Model Signature for MLflow
# ─────────────────────────────────────────────
def get_mlflow_signature():
    """
    MLflow model signature for validation at serving time.
    Input: image (float32 tensor) + metadata (float32 tensor)
    Output: class probabilities
    """
    try:
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec
        import numpy as np

        input_schema = Schema([
            TensorSpec(np.dtype("float32"), (-1, 3, 224, 224), "image"),
            TensorSpec(np.dtype("float32"), (-1, 5),           "metadata"),
        ])
        output_schema = Schema([
            TensorSpec(np.dtype("float32"), (-1, 7), "class_probabilities"),
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)
    except Exception as e:
        log.warning("Could not build MLflow signature: %s", e)
        return None
