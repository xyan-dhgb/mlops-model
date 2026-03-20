"""
XRAI Explainer for EfficientNet-B3 Image Branch
Reference: MDPI Cosmetics 2025 — EfficientNet-B3 + XRAI for skin cancer
Library: saliency (Google Research) — pip install saliency

Usage:
    explainer = XRAIExplainer(model)
    heatmap = explainer.explain(image_tensor, metadata_tensor)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────
# Grad-CAM (fallback + baseline comparison)
# ─────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM on the last conv block of EfficientNet-B3.
    Used as baseline comparison against XRAI per MDPI 2025 findings.
    """

    def __init__(self, model, target_layer_name: str = "features.8"):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_layer(target_layer_name)

    def _hook_layer(self, layer_name: str):
        layer = dict(self.model.image_branch.backbone.named_modules()).get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in EfficientNet-B3.")

        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    @torch.enable_grad()
    def generate(
        self,
        image: torch.Tensor,          # (1, 3, H, W)
        metadata: torch.Tensor,        # (1, 5)
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        image = image.requires_grad_(True)

        logits = self.model(image, metadata)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Weight activations by global-average-pooled gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


# ─────────────────────────────────────────────
# XRAI Explainer (Google saliency library)
# ─────────────────────────────────────────────
class XRAIExplainer:
    """
    XRAI (eXplanation with Ranked Area Insertions) for dermoscopic images.
    Produces region-based attributions via superpixel ranking — more coherent
    than pixel-level Grad-CAM for spatial lesion structures (MDPI 2025).

    Requires: pip install saliency
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self._xrai = None
        self._load_saliency()

    def _load_saliency(self):
        try:
            import saliency.core as saliency
            self._saliency = saliency
            self._xrai = saliency.XRAI()
            print("XRAI loaded via saliency library.")
        except ImportError:
            print(
                "saliency not installed. Run: pip install saliency\n"
                "Falling back to Grad-CAM for explanations."
            )
            self._saliency = None

    def _call_model_function(self, images_np: np.ndarray, call_model_args: dict):
        """
        Adapter: saliency library calls this with numpy images.
        Converts to torch, runs forward + backward, returns gradients.
        """
        images_tensor = torch.tensor(images_np, dtype=torch.float32).to(self.device)
        images_tensor.requires_grad_(True)
        metadata = call_model_args["metadata"].to(self.device)

        logits = self.model(images_tensor, metadata)
        target_class = call_model_args["target_class"]
        score = logits[:, target_class].sum()

        self.model.zero_grad()
        score.backward()
        gradients = images_tensor.grad.detach().cpu().numpy()
        return gradients

    @torch.enable_grad()
    def explain(
        self,
        image: torch.Tensor,       # (1, 3, 224, 224) — normalized
        metadata: torch.Tensor,    # (1, 5)
        target_class: Optional[int] = None,
        xrai_fast: bool = True,    # faster approximation (recommended for serving)
    ) -> np.ndarray:
        """
        Generate XRAI attribution map.
        Returns: (H, W) float32 array in [0, 1], higher = more important region.
        """
        self.model.eval()
        image = image.to(self.device)
        metadata = metadata.to(self.device)

        # Determine target class
        with torch.no_grad():
            logits = self.model(image, metadata)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        if self._xrai is None:
            # Fallback to Grad-CAM if saliency not available
            gradcam = GradCAM(self.model)
            return gradcam.generate(image, metadata, target_class)

        # XRAI requires (H, W, C) float32 in [0, 1]
        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Un-normalize from ImageNet stats for XRAI superpixel segmentation
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_vis = np.clip(img_np * std + mean, 0, 1).astype(np.float32)

        call_model_args = {"metadata": metadata, "target_class": target_class}

        xrai_params = self._saliency.XRAIParameters()
        xrai_params.algorithm = "fast" if xrai_fast else "full"

        attr = self._xrai.GetMask(
            img_vis,
            self._call_model_function,
            call_model_args=call_model_args,
            extra_parameters=xrai_params,
        )
        # Normalize
        if attr.max() > attr.min():
            attr = (attr - attr.min()) / (attr.max() - attr.min())
        return attr.astype(np.float32)

    def visualize(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
        class_names: Optional[list] = None,
    ) -> plt.Figure:
        """
        Side-by-side: original image | XRAI heatmap overlay.
        """
        attr = self.explain(image, metadata, target_class)

        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_vis = np.clip(img_np * std + mean, 0, 1)

        with torch.no_grad():
            logits = self.model(image.to(self.device), metadata.to(self.device))
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred   = probs.argmax()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(img_vis)
        axes[0].set_title("Original image", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(img_vis)
        heatmap = cm.jet(attr)[:, :, :3]
        axes[1].imshow(heatmap, alpha=0.45)
        label = class_names[pred] if class_names else str(pred)
        axes[1].set_title(f"XRAI — pred: {label} ({probs[pred]:.2%})", fontsize=12)
        axes[1].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
