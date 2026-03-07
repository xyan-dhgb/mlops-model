"""
tests/unit/test_multimodal_model.py
Unit tests cho kiến trúc Multimodal AI Model
(ResNet50 image branch + Dense tabular branch + Late Fusion)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.models.multimodal_model import (
    MultimodalCancerClassifier,
    ImageBranch,
    TabularBranch,
    FusionLayer,
    NUM_CLASSES,
    NUM_TABULAR_FEATURES,
)


# ══════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════
BATCH_SIZE = 4
IMG_H, IMG_W = 224, 224
N_TABULAR    = NUM_TABULAR_FEATURES  # 20

@pytest.fixture
def dummy_images():
    """Batch ảnh giả (B, C, H, W) — normalized [0,1]"""
    return torch.randn(BATCH_SIZE, 3, IMG_H, IMG_W)


@pytest.fixture
def dummy_tabular():
    """Batch tabular features giả (B, N_TABULAR)"""
    return torch.randn(BATCH_SIZE, N_TABULAR)


@pytest.fixture
def model():
    """Model với pretrained=False để test nhanh (không download weights)"""
    return MultimodalCancerClassifier(pretrained=False)


# ══════════════════════════════════════════════════════════
# TEST: ImageBranch
# ══════════════════════════════════════════════════════════
class TestImageBranch:
    def test_output_shape(self, dummy_images):
        branch = ImageBranch(pretrained=False)
        out = branch(dummy_images)
        # ImageBranch phải output vector 32-dim mỗi sample
        assert out.shape == (BATCH_SIZE, 32), f"Got {out.shape}"

    def test_output_is_float(self, dummy_images):
        branch = ImageBranch(pretrained=False)
        out = branch(dummy_images)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, dummy_images):
        branch = ImageBranch(pretrained=False)
        out = branch(dummy_images)
        assert not torch.isnan(out).any(), "NaN trong ImageBranch output"

    def test_resnet50_backbone_frozen(self):
        """Khi freeze=True, backbone weights không được update"""
        branch = ImageBranch(pretrained=False, freeze_backbone=True)
        for name, param in branch.named_parameters():
            if "backbone" in name:
                assert not param.requires_grad, \
                    f"Backbone param {name} phải bị freeze"

    def test_resnet50_backbone_trainable(self):
        branch = ImageBranch(pretrained=False, freeze_backbone=False)
        backbone_params = [p for n, p in branch.named_parameters()
                           if "backbone" in n]
        assert any(p.requires_grad for p in backbone_params)

    def test_dropout_disabled_in_eval_mode(self, dummy_images):
        """Eval mode: 2 forward passes phải cho kết quả giống nhau"""
        branch = ImageBranch(pretrained=False)
        branch.eval()
        with torch.no_grad():
            out1 = branch(dummy_images)
            out2 = branch(dummy_images)
        assert torch.allclose(out1, out2), "Eval mode không deterministic"

    def test_batch_size_1(self):
        """Edge case: batch size = 1"""
        branch = ImageBranch(pretrained=False)
        img = torch.randn(1, 3, 224, 224)
        out = branch(img)
        assert out.shape == (1, 32)


# ══════════════════════════════════════════════════════════
# TEST: TabularBranch
# ══════════════════════════════════════════════════════════
class TestTabularBranch:
    def test_output_shape(self, dummy_tabular):
        branch = TabularBranch(input_dim=N_TABULAR)
        out = branch(dummy_tabular)
        # TabularBranch phải output 32-dim (cùng chiều với ImageBranch)
        assert out.shape == (BATCH_SIZE, 32), f"Got {out.shape}"

    def test_no_nan_in_output(self, dummy_tabular):
        branch = TabularBranch(input_dim=N_TABULAR)
        out = branch(dummy_tabular)
        assert not torch.isnan(out).any()

    def test_accepts_correct_input_dim(self):
        branch = TabularBranch(input_dim=20)
        x = torch.randn(4, 20)
        out = branch(x)
        assert out.shape[1] == 32

    def test_wrong_input_dim_raises(self):
        branch = TabularBranch(input_dim=20)
        x = torch.randn(4, 15)  # Wrong dim
        with pytest.raises(RuntimeError):
            branch(x)

    def test_eval_deterministic(self, dummy_tabular):
        branch = TabularBranch(input_dim=N_TABULAR)
        branch.eval()
        with torch.no_grad():
            out1 = branch(dummy_tabular)
            out2 = branch(dummy_tabular)
        assert torch.allclose(out1, out2)


# ══════════════════════════════════════════════════════════
# TEST: FusionLayer
# ══════════════════════════════════════════════════════════
class TestFusionLayer:
    def test_output_shape_is_num_classes(self):
        fusion = FusionLayer(input_dim=64)  # 32+32 concatenated
        x = torch.randn(BATCH_SIZE, 64)
        out = fusion(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_is_logits_not_softmax(self):
        """FusionLayer phải return logits (raw), không phải probabilities"""
        fusion = FusionLayer(input_dim=64)
        x = torch.randn(BATCH_SIZE, 64)
        out = fusion(x)
        # Nếu là softmax → tổng mỗi row phải = 1.0 (với tolerance nhỏ)
        # Logits thì không nhất thiết như vậy
        row_sums = out.sum(dim=1)
        # Kiểm tra ít nhất 1 row không bằng 1 (logits)
        is_all_softmax = torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=0.01)
        assert not is_all_softmax, "FusionLayer nên return logits, không phải softmax"


# ══════════════════════════════════════════════════════════
# TEST: MultimodalCancerClassifier (end-to-end)
# ══════════════════════════════════════════════════════════
class TestMultimodalCancerClassifier:
    def test_forward_output_shape(self, model, dummy_images, dummy_tabular):
        logits = model(dummy_images, dummy_tabular)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES), \
            f"Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {logits.shape}"

    def test_output_6_classes(self, model, dummy_images, dummy_tabular):
        logits = model(dummy_images, dummy_tabular)
        assert logits.shape[1] == 6  # ACK, BCC, MEL, NEV, SCC, SEK

    def test_no_nan_in_forward(self, model, dummy_images, dummy_tabular):
        logits = model(dummy_images, dummy_tabular)
        assert not torch.isnan(logits).any()

    def test_gradients_flow(self, model, dummy_images, dummy_tabular):
        """Gradient phải flow từ loss đến parameters"""
        model.train()
        logits = model(dummy_images, dummy_tabular)
        target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Kiểm tra ít nhất 1 param có gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "Không có gradient flow"

    def test_eval_mode_deterministic(self, model, dummy_images, dummy_tabular):
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_images, dummy_tabular)
            out2 = model(dummy_images, dummy_tabular)
        assert torch.allclose(out1, out2)

    def test_predict_returns_class_indices(self, model, dummy_images, dummy_tabular):
        """predict() phải trả về class indices [0-5]"""
        model.eval()
        with torch.no_grad():
            preds = model.predict(dummy_images, dummy_tabular)
        assert preds.shape == (BATCH_SIZE,)
        assert (preds >= 0).all() and (preds < NUM_CLASSES).all()

    def test_predict_proba_sums_to_one(self, model, dummy_images, dummy_tabular):
        """predict_proba() phải trả về probabilities tổng = 1"""
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(dummy_images, dummy_tabular)
        assert probs.shape == (BATCH_SIZE, NUM_CLASSES)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_count_parameters(self, model):
        """Model phải có số params hợp lý (không quá nhỏ hay quá lớn)"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # ResNet50 có ~23M params; model tổng phải > 1M
        assert total > 1_000_000, f"Model quá nhỏ: {total:,} params"
        print(f"\nModel params: {total:,} total, {trainable:,} trainable")

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_various_batch_sizes(self, batch_size):
        """Model phải chạy được với batch size bất kỳ"""
        model = MultimodalCancerClassifier(pretrained=False)
        imgs  = torch.randn(batch_size, 3, 224, 224)
        tabs  = torch.randn(batch_size, N_TABULAR)
        out   = model(imgs, tabs)
        assert out.shape == (batch_size, NUM_CLASSES)
