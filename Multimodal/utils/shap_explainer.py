"""
SHAP Explainer for Metadata MLP Branch
Reference: SkinSage PMC 2024 — Multimodal AI + SHAP per-feature explanation

Explains contribution of age, sex, localization to each prediction.
Example insight: 'age > 60 + localization: back → ↑ melanoma probability'

Usage:
    explainer = SHAPMetadataExplainer(model, background_metadata)
    shap_vals = explainer.explain(metadata_tensor)
    explainer.plot_waterfall(shap_vals, feature_values, class_idx=0)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

FEATURE_NAMES = ["age_norm", "is_male", "high_risk_site", "age_bucket", "site_encoded"]
CLASS_NAMES   = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


# ─────────────────────────────────────────────
# Metadata-only forward wrapper (for SHAP)
# ─────────────────────────────────────────────
class MetadataOnlyWrapper(torch.nn.Module):
    """
    Wraps the full model, freezing a fixed image embedding.
    SHAP only perturbs metadata inputs — image stays constant.
    This isolates the metadata branch's contribution.
    """

    def __init__(self, model, fixed_image: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer(
            "fixed_image",
            fixed_image.expand(1, -1, -1, -1)   # (1, 3, H, W)
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        B = metadata.shape[0]
        imgs = self.fixed_image.expand(B, -1, -1, -1)
        return self.model(imgs, metadata)


# ─────────────────────────────────────────────
# SHAP Explainer
# ─────────────────────────────────────────────
class SHAPMetadataExplainer:
    """
    DeepExplainer SHAP for the metadata MLP branch.
    Background dataset = representative sample from training set.

    Args:
        model              : MultimodalSkinClassifier
        background_metadata: (N, 5) tensor — training set background for SHAP
        background_image   : (1, 3, 224, 224) tensor — representative image
                             (held fixed while metadata is perturbed)
        device             : "cuda" or "cpu"
    """

    def __init__(
        self,
        model,
        background_metadata: torch.Tensor,
        background_image: torch.Tensor,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        # Build metadata-only wrapper
        bg_img = background_image.to(self.device)
        self.wrapper = MetadataOnlyWrapper(self.model, bg_img).to(self.device)

        self._shap_explainer = None
        self.background = background_metadata.to(self.device)
        self._load_explainer()

    def _load_explainer(self):
        try:
            import shap
            self._shap = shap
            self._shap_explainer = shap.DeepExplainer(
                self.wrapper,
                self.background,
            )
            print(f"SHAP DeepExplainer ready. Background size: {self.background.shape[0]}")
        except ImportError:
            print("shap not installed. Run: pip install shap")
            self._shap = None

    def explain(
        self,
        metadata: torch.Tensor,   # (N, 5)
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute SHAP values for a batch of metadata inputs.

        Returns:
            If class_idx given  : (N, 5) SHAP values for that class
            If class_idx is None: (7, N, 5) for all classes
        """
        if self._shap_explainer is None:
            raise RuntimeError("SHAP not available. pip install shap")

        metadata = metadata.to(self.device)
        shap_values = self._shap_explainer.shap_values(metadata)
        # shap_values: list of 7 arrays, each (N, 5)
        shap_array = np.stack(shap_values, axis=0)  # (7, N, 5)

        if class_idx is not None:
            return shap_array[class_idx]             # (N, 5)
        return shap_array                            # (7, N, 5)

    # ── Visualization ──────────────────────────────────────────────────────

    def plot_waterfall(
        self,
        shap_values: np.ndarray,     # (5,) for one sample + one class
        feature_values: np.ndarray,  # (5,) actual input values
        class_idx: int = 0,
        base_value: float = 0.0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Waterfall chart: one sample, one class.
        Shows how each feature pushes probability up or down from base.
        """
        n_features = len(FEATURE_NAMES)
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]

        colors = ["#E8593C" if v > 0 else "#3B8BD4" for v in shap_values[sorted_idx]]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(
            range(n_features),
            shap_values[sorted_idx],
            color=colors,
            edgecolor="none",
            height=0.55,
        )

        # Feature labels with actual values
        labels = []
        for i in sorted_idx:
            name = FEATURE_NAMES[i]
            val  = feature_values[i]
            labels.append(f"{name} = {val:.2f}")
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(labels, fontsize=10)

        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
        ax.set_title(
            f"SHAP — {CLASS_NAMES[class_idx]} prediction\n"
            f"Base value: {base_value:.3f}  |  Sum: {shap_values.sum():.3f}",
            fontsize=11,
        )

        red_patch  = mpatches.Patch(color="#E8593C", label="Increases probability")
        blue_patch = mpatches.Patch(color="#3B8BD4", label="Decreases probability")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="lower right")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_summary(
        self,
        shap_values: np.ndarray,  # (N, 5) for one class
        class_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Summary beeswarm plot — overall feature importance across dataset.
        Requires shap library for beeswarm rendering.
        """
        if self._shap is None:
            raise RuntimeError("shap not installed.")

        fig, ax = plt.subplots(figsize=(8, 4))
        self._shap.summary_plot(
            shap_values,
            feature_names=FEATURE_NAMES,
            plot_type="dot",
            show=False,
            color_bar=True,
        )
        ax.set_title(f"SHAP summary — {CLASS_NAMES[class_idx]}", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def top_features(
        self,
        shap_values: np.ndarray,  # (5,) one sample
        top_k: int = 3,
    ) -> list:
        """
        Return top-k (feature_name, shap_value, direction) for API response.
        Used when serving: attach human-readable explanation to inference output.
        """
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:top_k]
        results = []
        for i in sorted_idx:
            results.append({
                "feature": FEATURE_NAMES[i],
                "shap_value": float(shap_values[i]),
                "direction": "increases" if shap_values[i] > 0 else "decreases",
            })
        return results


# ─────────────────────────────────────────────
# Unified XAI Runner (image + tabular together)
# ─────────────────────────────────────────────
class MultimodalXAIRunner:
    """
    Convenience wrapper to run XRAI + SHAP in one call.
    Returns a dict suitable for attaching to KServe inference response.
    """

    def __init__(
        self,
        model,
        background_metadata: torch.Tensor,
        background_image: torch.Tensor,
        device: str = "cuda",
    ):
        from utils.xrai_explainer import XRAIExplainer
        self.xrai  = XRAIExplainer(model, device)
        self.shap  = SHAPMetadataExplainer(
            model, background_metadata, background_image, device
        )
        self.device = device

    def explain(
        self,
        image: torch.Tensor,      # (1, 3, 224, 224)
        metadata: torch.Tensor,   # (1, 5)
        target_class: Optional[int] = None,
        top_k_shap: int = 3,
    ) -> dict:
        """
        Full XAI explanation for one sample.

        Returns dict with:
          - predicted_class (str)
          - confidence (float)
          - xrai_heatmap (np.ndarray H×W)
          - shap_top_features (list of dicts)
        """
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            logits = self.xrai.model(image.to(dev), metadata.to(dev))
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_class = int(probs.argmax())
        if target_class is None:
            target_class = pred_class

        xrai_map = self.xrai.explain(image, metadata, target_class)
        shap_vals = self.shap.explain(metadata, class_idx=target_class).squeeze()
        top_features = self.shap.top_features(shap_vals, top_k=top_k_shap)

        return {
            "predicted_class": CLASS_NAMES[pred_class],
            "confidence": float(probs[pred_class]),
            "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "xrai_heatmap": xrai_map,
            "shap_top_features": top_features,
        }
