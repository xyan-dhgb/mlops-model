"""
Unit Tests — Model Architecture
Tests: forward pass shapes, FocalLoss, parameter counts, gradient flow
Run: pytest tests/test_model.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from Multimodal.models.multimodal_model import (
    MultimodalSkinClassifier,
    MetadataMLP,
    ImageBackbone,
    FocalLoss,
)

BATCH_SIZE   = 4
IMAGE_SIZE   = 224
META_DIM     = 5
NUM_CLASSES  = 7
DEVICE       = "cpu"   # Use CPU for tests (no GPU required in CI)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    m = MultimodalSkinClassifier(
        num_classes=NUM_CLASSES,
        metadata_input_dim=META_DIM,
        pretrained=False,          # faster — no download in CI
    )
    return m.eval()


@pytest.fixture
def dummy_batch():
    imgs = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta = torch.randn(BATCH_SIZE, META_DIM)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return imgs, meta, labels


# ─────────────────────────────────────────────
# MetadataMLP Tests
# ─────────────────────────────────────────────
class TestMetadataMLP:
    def test_output_shape(self):
        mlp = MetadataMLP(input_dim=META_DIM, hidden_dim=64)
        x   = torch.randn(BATCH_SIZE, META_DIM)
        out = mlp(x)
        assert out.shape == (BATCH_SIZE, 64)

    def test_batch_norm_in_eval(self):
        mlp = MetadataMLP(input_dim=META_DIM, hidden_dim=64).eval()
        x   = torch.randn(BATCH_SIZE, META_DIM)
        with torch.no_grad():
            out = mlp(x)
        assert out.shape == (BATCH_SIZE, 64)

    def test_single_sample_eval(self):
        """Single sample must work in eval mode (BN uses running stats)."""
        mlp = MetadataMLP(input_dim=META_DIM, hidden_dim=64).eval()
        x   = torch.randn(1, META_DIM)
        with torch.no_grad():
            out = mlp(x)
        assert out.shape == (1, 64)

    def test_no_nan_output(self):
        mlp = MetadataMLP(input_dim=META_DIM, hidden_dim=64).eval()
        x   = torch.randn(BATCH_SIZE, META_DIM)
        with torch.no_grad():
            out = mlp(x)
        assert not torch.isnan(out).any()


# ─────────────────────────────────────────────
# ImageBackbone Tests
# ─────────────────────────────────────────────
class TestImageBackbone:
    def test_output_feature_dim(self):
        backbone = ImageBackbone(pretrained=False).eval()
        x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (BATCH_SIZE, backbone.feature_dim)

    def test_feature_dim_is_1536(self):
        backbone = ImageBackbone(pretrained=False)
        assert backbone.feature_dim == 1536

    def test_no_nan_output(self):
        backbone = ImageBackbone(pretrained=False).eval()
        x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = backbone(x)
        assert not torch.isnan(out).any()


# ─────────────────────────────────────────────
# MultimodalSkinClassifier Tests
# ─────────────────────────────────────────────
class TestMultimodalModel:
    def test_forward_output_shape(self, model, dummy_batch):
        imgs, meta, _ = dummy_batch
        with torch.no_grad():
            out = model(imgs, meta)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_is_logits_not_probs(self, model, dummy_batch):
        """Output should be raw logits, not softmax probabilities."""
        imgs, meta, _ = dummy_batch
        with torch.no_grad():
            out = model(imgs, meta)
        # Logits can be negative; probabilities cannot
        assert out.min().item() < 0 or out.max().item() > 1

    def test_no_nan_output(self, model, dummy_batch):
        imgs, meta, _ = dummy_batch
        with torch.no_grad():
            out = model(imgs, meta)
        assert not torch.isnan(out).any()

    def test_single_sample_eval(self, model):
        img  = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        meta = torch.randn(1, META_DIM)
        model.eval()
        with torch.no_grad():
            out = model(img, meta)
        assert out.shape == (1, NUM_CLASSES)

    def test_parameter_count_reasonable(self, model):
        """EfficientNet-B3 + MLP should be ~12-15M parameters."""
        total = sum(p.numel() for p in model.parameters())
        assert 10_000_000 < total < 20_000_000, f"Unexpected param count: {total}"

    def test_gradient_flows_to_both_branches(self, model, dummy_batch):
        """Backprop should reach both image and metadata branch."""
        model.train()
        imgs, meta, labels = dummy_batch

        logits = model(imgs, meta)
        loss   = logits.sum()
        loss.backward()

        # Check image branch has gradients
        img_grad = model.image_branch.backbone.features[0][0].weight.grad
        assert img_grad is not None
        assert not torch.isnan(img_grad).any()

        # Check metadata branch has gradients
        meta_grad = model.metadata_branch.mlp[0].weight.grad
        assert meta_grad is not None
        assert not torch.isnan(meta_grad).any()

        model.eval()  # restore

    def test_different_classes_get_different_logits(self, model, dummy_batch):
        """All logits should not be identical (model is not degenerate)."""
        imgs, meta, _ = dummy_batch
        with torch.no_grad():
            out = model(imgs, meta)
        # At least some variation across classes
        assert out[0].std().item() > 0

    def test_get_image_features_shape(self, model, dummy_batch):
        imgs, _, _ = dummy_batch
        with torch.no_grad():
            feats = model.get_image_features(imgs)
        assert feats.shape == (BATCH_SIZE, 1536)


# ─────────────────────────────────────────────
# FocalLoss Tests
# ─────────────────────────────────────────────
class TestFocalLoss:
    def test_loss_is_scalar(self, dummy_batch):
        imgs, meta, labels = dummy_batch
        model = MultimodalSkinClassifier(num_classes=NUM_CLASSES, pretrained=False)
        with torch.no_grad():
            logits = model(imgs, meta)
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, labels)
        assert loss.shape == ()

    def test_loss_is_positive(self, dummy_batch):
        imgs, meta, labels = dummy_batch
        model = MultimodalSkinClassifier(num_classes=NUM_CLASSES, pretrained=False)
        with torch.no_grad():
            logits = model(imgs, meta)
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_perfect_prediction_lower_loss(self):
        """High-confidence correct prediction → lower focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        # High-confidence correct prediction
        logits_good = torch.tensor([[10.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
        labels = torch.tensor([0])
        loss_good = loss_fn(logits_good, labels).item()

        # Low-confidence correct prediction
        logits_bad = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        loss_bad = loss_fn(logits_bad, labels).item()

        assert loss_good < loss_bad

    def test_class_weights_applied(self, dummy_batch):
        imgs, meta, labels = dummy_batch
        model = MultimodalSkinClassifier(num_classes=NUM_CLASSES, pretrained=False)
        with torch.no_grad():
            logits = model(imgs, meta)

        weights_high = torch.ones(NUM_CLASSES) * 5.0
        weights_low  = torch.ones(NUM_CLASSES) * 1.0

        loss_high = FocalLoss(alpha=weights_high, gamma=2.0)(logits, labels).item()
        loss_low  = FocalLoss(alpha=weights_low,  gamma=2.0)(logits, labels).item()
        assert loss_high > loss_low

    def test_no_nan_loss(self, dummy_batch):
        imgs, meta, labels = dummy_batch
        model = MultimodalSkinClassifier(num_classes=NUM_CLASSES, pretrained=False)
        with torch.no_grad():
            logits = model(imgs, meta)
        loss = FocalLoss(gamma=2.0)(logits, labels)
        assert not torch.isnan(loss)
