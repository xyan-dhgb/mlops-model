"""
Multimodal Skin Cancer Detection — ISIC 2024
Kien truc: EfficientNet-B3 (anh) + MetadataMLP (9-dim) -> Fusion -> Binary output
Task: Binary classification (malignant=1 / benign=0)
Loss: BinaryFocalLoss + pos_weight (BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Binary Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss cho binary classification voi BCEWithLogitsLoss.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma     : Tham so focusing (2.0 theo RetinaNet)
        pos_weight: Trong so duong tinh (n_neg/n_pos cho ISIC 2024 ~33)
        alpha     : Trong so lop duong tinh [0,1] (None = 0.25 theo RetinaNet)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B,) hoac (B, 1) — raw output chua qua sigmoid
            targets: (B,) float32 — 0.0 hoac 1.0
        """
        logits  = logits.squeeze(1) if logits.dim() == 2 else logits
        targets = targets.float()

        # BCE co pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # Focal weighting
        p_t = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)

        # Alpha weighting
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device),
        )

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metadata MLP Branch — 9-dim input (ISIC 2024)
# ─────────────────────────────────────────────────────────────────────────────
class MetadataMLP(nn.Module):
    """
    MLP 3 tang cho metadata ISIC 2024.
    Input : 9 chieu [age, sex, site, age_bucket, high_risk_site,
                     tbp_nevi_conf, tbp_color_asym, tbp_size, tbp_border]
    Output: 64 chieu embedding
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# Image Backbone — EfficientNet-B3
# ─────────────────────────────────────────────────────────────────────────────
class ImageBackbone(nn.Module):
    """
    EfficientNet-B3 pretrained ImageNet.
    Output: 1536-dim feature vector (bo classifier goc).
    """

    def __init__(self, pretrained: bool = True, freeze_bn: bool = False):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        self.feature_dim = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Identity()

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 1536)


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal Binary Classifier — ISIC 2024
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalSkinClassifier(nn.Module):
    """
    Late fusion: concat(image_feat 1536 + meta_feat 64) -> FC(512) -> 1 logit

    Dau ra la LOGIT don (chua qua sigmoid):
      - Dung voi BinaryFocalLoss (BCEWithLogitsLoss ben trong)
      - Khi inference: torch.sigmoid(logit) > 0.5 -> malignant

    So sanh voi phien ban ISIC 2019:
      - num_classes: 7 -> 1
      - metadata_input_dim: 5 -> 9
      - FocalLoss 7-class -> BinaryFocalLoss
    """

    def __init__(
        self,
        num_classes: int = 1,          # binary: luon la 1
        metadata_input_dim: int = 9,   # ISIC 2024: 9 features
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()
        assert num_classes == 1, "ISIC 2024 la binary task, num_classes phai bang 1"

        self.image_branch    = ImageBackbone(pretrained=pretrained, freeze_bn=freeze_bn)
        self.metadata_branch = MetadataMLP(input_dim=metadata_input_dim, hidden_dim=64)

        fusion_input = self.image_branch.feature_dim + 64  # 1600

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),  # binary logit
        )

    def forward(
        self,
        images: torch.Tensor,    # (B, 3, 224, 224)
        metadata: torch.Tensor,  # (B, 9)
    ) -> torch.Tensor:
        img_feat  = self.image_branch(images)       # (B, 1536)
        meta_feat = self.metadata_branch(metadata)  # (B, 64)
        fused     = torch.cat([img_feat, meta_feat], dim=1)  # (B, 1600)
        return self.fusion(fused).squeeze(1)        # (B,) — logit don

    def predict_proba(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Tra ve xac suat malignant sau sigmoid. Dung khi inference."""
        with torch.no_grad():
            logits = self.forward(images, metadata)
        return torch.sigmoid(logits)  # (B,) trong [0, 1]

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Lay feature map cho XRAI/Grad-CAM."""
        return self.image_branch(images)


# ─────────────────────────────────────────────────────────────────────────────
# Alias giu tuong thich voi cac import cu
# ─────────────────────────────────────────────────────────────────────────────
FocalLoss = BinaryFocalLoss


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MultimodalSkinClassifier(num_classes=1, metadata_input_dim=9)
    B     = 4
    imgs  = torch.randn(B, 3, 224, 224)
    meta  = torch.randn(B, 9)
    out   = model(imgs, meta)
    print(f"Output shape : {out.shape}")   # (4,) — logit don
    print(f"Proba sample : {model.predict_proba(imgs, meta)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total / 1e6:.1f}M")

    labels   = torch.randint(0, 2, (B,)).float()
    loss_fn  = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    loss     = loss_fn(out, labels)
    print(f"BinaryFocalLoss: {loss.item():.4f}")
    print("OK")
