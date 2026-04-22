"""
utils/xai.py
=============
Explainability (XAI) tools for the ISIC 2024 multimodal model.

Classes
-------
GradCAM               : Gradient-weighted Class Activation Maps for the image branch
SHAPTabularExplainer  : DeepExplainer-based SHAP for the tabular branch
MultimodalXAIRunner   : Combines both and renders a unified explanation figure

Usage
-----
    runner = MultimodalXAIRunner(model, background_tabular, background_image,
                                 feature_names, last_conv_layer="top_activation")
    result = runner.explain(tab_sample, img_sample, visualize=True)
    # result keys: "heatmap", "overlay", "shap_values", "pred_class", "confidence"
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


# ── GRAD-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for the image branch of the multimodal model.

    Parameters
    ----------
    model           : fitted Keras model with named inputs
                      ("image_input", "tabular_input")
    last_conv_layer : name of the last convolutional / activation layer
                      (EfficientNetB3 → "top_activation")
    """

    def __init__(self, model, last_conv_layer: str = "top_activation"):
        self.model          = model
        self.last_conv_layer = last_conv_layer
        self._grad_model    = self._build_grad_model()

    def _build_grad_model(self):
        """Build a model that outputs (conv_activations, predictions)."""
        conv_layer = self.model.get_layer(self.last_conv_layer)
        return tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )

    def compute_heatmap(self,
                        img_input: np.ndarray,
                        tab_input: np.ndarray,
                        class_idx: int = 0) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap for a single sample.

        Parameters
        ----------
        img_input : (1, H, W, 3) float32
        tab_input : (1, n_features) float32
        class_idx : output neuron index (0 for binary sigmoid)

        Returns
        -------
        heatmap : (H', W') float32 normalised to [0, 1]
        """
        img_tensor = tf.cast(img_input, tf.float32)
        tab_tensor = tf.cast(tab_input, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = self._grad_model(
                [img_tensor, tab_tensor], training=False
            )
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling over spatial dims
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap)

        # Normalise
        heatmap = heatmap.numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap.astype(np.float32)

    @staticmethod
    def overlay_heatmap(heatmap: np.ndarray,
                        img_array: np.ndarray,
                        alpha: float = 0.4,
                        colormap: int = None) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap onto the original image.

        Parameters
        ----------
        heatmap   : (H', W') float32 in [0, 1]
        img_array : (H, W, 3) float32 original image
        alpha     : blending strength

        Returns
        -------
        overlay : (H, W, 3) uint8
        """
        import cv2
        h, w = img_array.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
        overlay   = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay


# ── SHAP TABULAR EXPLAINER ────────────────────────────────────────────────────

class SHAPTabularExplainer:
    """
    SHAP DeepExplainer for the tabular branch.

    Parameters
    ----------
    model             : fitted Keras multimodal model
    background_tabular: (N, F) background dataset for SHAP reference
    background_image  : (H, W, 3) a representative background image
    """

    def __init__(self, model, background_tabular: np.ndarray,
                 background_image: np.ndarray):
        try:
            import shap
            self._shap = shap
        except ImportError:
            raise ImportError("Install shap:  pip install shap")

        # Build a wrapper that accepts only tabular input (image held fixed)
        self._model   = model
        self._bg_img  = background_image[np.newaxis, ...].astype(np.float32)
        self._bg_tab  = background_tabular.astype(np.float32)

        self.explainer = shap.DeepExplainer(
            model=self._model,
            data=[
                np.repeat(self._bg_img, len(self._bg_tab), axis=0),
                self._bg_tab,
            ],
        )

    def explain(self, tabular_samples: np.ndarray) -> list:
        """
        Compute SHAP values for one or more tabular samples.

        Parameters
        ----------
        tabular_samples : (N, F) or (F,) float32

        Returns
        -------
        shap_values : list[np.ndarray]  (DeepExplainer output format)
        """
        if tabular_samples.ndim == 1:
            tabular_samples = tabular_samples[np.newaxis, :]

        n = len(tabular_samples)
        img_repeated = np.repeat(self._bg_img, n, axis=0)

        shap_values = self.explainer.shap_values(
            [img_repeated, tabular_samples.astype(np.float32)]
        )
        return shap_values


# ── COMBINED XAI RUNNER ───────────────────────────────────────────────────────

class MultimodalXAIRunner:
    """
    Combines Grad-CAM (image) and SHAP (tabular) into one unified explanation.

    Parameters
    ----------
    model             : fitted Keras multimodal model
    background_tabular: (N, F) background for SHAP
    background_image  : (H, W, 3) single representative image
    feature_names     : list of tabular feature names
    last_conv_layer   : Grad-CAM target layer name
    """

    def __init__(self,
                 model,
                 background_tabular: np.ndarray,
                 background_image:   np.ndarray,
                 feature_names:      list,
                 last_conv_layer:    str = "top_activation"):

        self.model         = model
        self.feature_names = feature_names
        self.gradcam       = GradCAM(model, last_conv_layer=last_conv_layer)
        self.shap_explainer = SHAPTabularExplainer(
            model, background_tabular, background_image
        )

    def explain(self,
                tab_sample:  np.ndarray,
                img_sample:  np.ndarray,
                visualize:   bool = True) -> dict:
        """
        Explain a single prediction.

        Parameters
        ----------
        tab_sample : (F,)      tabular features
        img_sample : (H, W, 3) image array
        visualize  : if True, render a Grad-CAM + SHAP waterfall figure

        Returns
        -------
        dict with keys: pred_class, confidence, heatmap, overlay, shap_values
        """
        tab_in = tab_sample[np.newaxis, ...].astype(np.float32)
        img_in = img_sample[np.newaxis, ...].astype(np.float32)

        # ── Prediction ───────────────────────────────────────────────────────
        pred = self.model.predict(
            {"image_input": img_in, "tabular_input": tab_in}, verbose=0
        )
        if pred.shape[-1] == 1:
            confidence = float(pred[0, 0])
            pred_class = int(confidence > 0.5)
        else:
            pred_class = int(np.argmax(pred[0]))
            confidence = float(np.max(pred[0]))

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        heatmap = self.gradcam.compute_heatmap(img_in, tab_in)
        overlay = self.gradcam.overlay_heatmap(heatmap, img_sample)

        # ── SHAP ─────────────────────────────────────────────────────────────
        shap_values = self.shap_explainer.explain(tab_sample)

        # ── Visualise ────────────────────────────────────────────────────────
        if visualize:
            self._render(img_sample, overlay, shap_values, tab_sample,
                         pred_class, confidence)

        return {
            "pred_class":  pred_class,
            "confidence":  confidence,
            "heatmap":     heatmap,
            "overlay":     overlay,
            "shap_values": shap_values,
        }

    def _render(self, img_sample, overlay, shap_values, tab_sample,
                pred_class, confidence):
        """Render a 2-panel figure: Grad-CAM left, SHAP waterfall right."""
        try:
            import shap
        except ImportError:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        label = "Malignant" if pred_class == 1 else "Benign"
        color = "red" if pred_class == 1 else "green"
        fig.suptitle(f"Prediction: {label}  (confidence {confidence:.2%})",
                     fontsize=14, fontweight="bold", color=color)

        # Left — Grad-CAM
        ax1.imshow(overlay)
        ax1.set_title("Grad-CAM  (image branch)", fontweight="bold")
        ax1.axis("off")

        # Right — SHAP waterfall
        if isinstance(shap_values, list) and len(shap_values) > 0:
            sv = shap_values[0]
            if sv.ndim == 2:
                sv = sv[0]             # (F,)
            elif sv.ndim == 3:
                sv = sv[0].flatten()

            expected = self.shap_explainer.explainer.expected_value
            if tf.is_tensor(expected):
                expected = expected.numpy()
            if isinstance(expected, np.ndarray):
                expected = float(expected.flat[0])

            expl = shap.Explanation(
                values=sv,
                base_values=expected,
                data=tab_sample,
                feature_names=self.feature_names,
            )
            plt.sca(ax2)
            shap.plots.waterfall(expl, max_display=10, show=False)
            ax2.set_title("SHAP Waterfall  (tabular branch)", fontweight="bold")

        plt.tight_layout()
        plt.show()

    # ── Convenience: global importance bar chart ──────────────────────────────

    def global_importance(self,
                          X_tab:    np.ndarray,
                          top_k:    int = 15,
                          n_sample: int = 50,
                          save_path: str | None = None):
        """
        Compute and plot mean |SHAP| across a sample of instances.

        Parameters
        ----------
        X_tab    : (N, F) tabular array from training set
        top_k    : number of top features to display
        n_sample : how many instances to compute SHAP for (keep ≤ 200)
        """
        indices  = np.random.choice(len(X_tab), min(n_sample, len(X_tab)),
                                    replace=False)
        X_sample = X_tab[indices]

        print(f"[global_importance] Computing SHAP on {len(X_sample)} samples…")
        shap_values = self.shap_explainer.explain(X_sample)

        sv = shap_values[0]       # (N, F) or (N, F, 1)
        if sv.ndim == 3:
            sv = sv[:, :, 0]

        mean_abs = np.mean(np.abs(sv), axis=0)   # (F,)
        sorted_idx = np.argsort(mean_abs)[::-1]
        top_idx    = sorted_idx[:top_k]

        plt.figure(figsize=(8, 6))
        plt.barh(
            [self.feature_names[i] for i in top_idx],
            mean_abs[top_idx],
        )
        plt.gca().invert_yaxis()
        plt.title(f"Global Feature Importance — Top {top_k} (Mean |SHAP|)",
                  fontweight="bold")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[global_importance] Saved → {save_path}")
        plt.show()

        return {
            "feature_names": [self.feature_names[i] for i in top_idx],
            "mean_abs_shap":  mean_abs[top_idx].tolist(),
        }
