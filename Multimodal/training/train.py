"""
Training pipeline for ISIC 2024 multimodal model.
Includes: class-weight balancing, callbacks, evaluation, GradCAM, SHAP.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import mlflow

from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_tab_train: np.ndarray,
    X_img_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_val: np.ndarray,
    X_img_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    checkpoint_path: str = "Multimodal/final/best_model.h5"
):
    """
    Train the multimodal model with class-weight balancing and callbacks.

    Returns:
        Keras History object.
    """
    # Balanced class weights (ISIC 2024 is heavily imbalanced)
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=5,
                      restore_best_weights=True, mode="max"),
        ModelCheckpoint(checkpoint_path, monitor="val_auc",
                        save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        {"image_input": X_img_train, "tabular_input": X_tab_train},
        y_train,
        validation_data=(
            {"image_input": X_img_val, "tabular_input": X_tab_val},
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_tab_test: np.ndarray,
    X_img_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder=None,
    threshold: float = 0.5
) -> tuple[float, float]:
    """
    Full evaluation: accuracy, AUC, pAUC, confusion matrix, classification report.

    Returns:
        (accuracy, auc_score)
    """
    preds = model.predict(
        {"image_input": X_img_test, "tabular_input": X_tab_test}, verbose=0
    )

    if preds.shape[-1] == 1:
        y_prob = preds.flatten()
        y_pred = (y_prob > threshold).astype(int)
    else:
        y_prob = preds[:, 1]
        y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    # Partial AUC at TPR ≥ 80 % (ISIC 2024 competition metric)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    mask = tpr >= 0.8
    pauc = auc(fpr[mask], tpr[mask]) / (1.0 - 0.8) if mask.sum() > 1 else float("nan")

    print(f"Accuracy : {acc:.4f}")
    print(f"AUC-ROC  : {auc_score:.4f}")
    print(f"pAUC@80% : {pauc:.4f}  ← ISIC 2024 competition metric")
    print(classification_report(y_test, y_pred,
                                target_names=["Benign (0)", "Malignant (1)"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm)

    return acc, auc_score


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    import seaborn as sns
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_training_history(history) -> None:
    """Plot accuracy, loss and AUC curves."""
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)
    n_plots = 2 + int("auc" in hist)
    plt.figure(figsize=(6 * n_plots, 5))

    for i, (key, label) in enumerate([("accuracy", "Accuracy"), ("loss", "Loss")]):
        plt.subplot(1, n_plots, i + 1)
        if key in hist:
            plt.plot(epochs, hist[key], "bo-", label=f"Train {label}")
        if f"val_{key}" in hist:
            plt.plot(epochs, hist[f"val_{key}"], "ro-", label=f"Val {label}")
        plt.title(label)
        plt.xlabel("Epochs")
        plt.legend()

    if "auc" in hist:
        plt.subplot(1, n_plots, 3)
        plt.plot(epochs, hist["auc"], "go-", label="Train AUC")
        if "val_auc" in hist:
            plt.plot(epochs, hist["val_auc"], "mo-", label="Val AUC")
        plt.title("AUC (ISIC 2024 metric)")
        plt.xlabel("Epochs")
        plt.legend()

    plt.tight_layout()
    plt.show()


# ── XAI ───────────────────────────────────────────────────────────────────────

class GradCAMExplainer:
    """Grad-CAM heatmap over the last convolutional layer."""

    def __init__(self, model, last_conv_layer_name: str):
        self.model = model
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

    def compute_heatmap(
        self,
        image: np.ndarray,
        tabular: np.ndarray,
        class_idx: int | None = None
    ) -> np.ndarray:
        """
        Args:
            image:   (1, H, W, 3) float32
            tabular: (1, F) float32
        Returns:
            2-D heatmap array, values in [0, 1].
        """
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model([image, tabular])
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        h, w = image.shape[:2]
        heatmap_u8 = cv2.applyColorMap(
            (cv2.resize(heatmap, (w, h)) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        base = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        return cv2.addWeighted(base, 1 - alpha, heatmap_u8, alpha, 0)


class SHAPMetadataExplainer:
    """SHAP DeepExplainer wrapper for the tabular branch."""

    def __init__(self, model, background_tabular: np.ndarray,
                 background_image: np.ndarray, feature_names: list[str]):
        import shap

        if background_image.ndim == 3:
            background_image = background_image[np.newaxis]

        bg_img_tensor = tf.convert_to_tensor(background_image, dtype=tf.float32)
        H, W, C = bg_img_tensor.shape[-3:]

        tab_input = tf.keras.Input(shape=background_tabular.shape[1:])

        def tile_image(x):
            batch = tf.shape(x)[0]
            return tf.tile(bg_img_tensor, [batch, 1, 1, 1])

        img_tiled = tf.keras.layers.Lambda(
            tile_image, output_shape=(H, W, C)
        )(tab_input)

        output = model({"image_input": img_tiled, "tabular_input": tab_input})
        self.wrapper_model = tf.keras.Model(inputs=tab_input, outputs=output)
        self.explainer = shap.DeepExplainer(self.wrapper_model, background_tabular)
        self.feature_names = feature_names

    def explain(self, X: np.ndarray):
        return self.explainer.shap_values(X)


class MultimodalXAIRunner:
    """Runs GradCAM + SHAP together and visualises the results."""

    def __init__(self, model, background_tabular, background_image,
                 feature_names, last_conv_layer: str):
        print("Initialising Multimodal XAI Runner...")
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = SHAPMetadataExplainer(
            model, background_tabular, background_image, feature_names
        )
        self.gradcam = GradCAMExplainer(model, last_conv_layer)
        print("XAI ready.")

    def explain(
        self,
        tabular_sample: np.ndarray,
        image_sample: np.ndarray,
        visualize: bool = True
    ) -> dict:
        tab_exp = tabular_sample[np.newaxis]
        img_exp = image_sample[np.newaxis]

        shap_values = self.shap_explainer.explain(tab_exp)
        heatmap = self.gradcam.compute_heatmap(img_exp, tab_exp)

        if visualize:
            self._visualize(image_sample, heatmap, shap_values, tabular_sample)

        return {"shap_values": shap_values, "heatmap": heatmap}

    def _visualize(self, image, heatmap, shap_values, tabular_sample):
        import shap as shap_lib

        overlay = self.gradcam.overlay_heatmap(heatmap, image)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].imshow(overlay)
        axes[0].set_title("Grad-CAM Heatmap")
        axes[0].axis("off")

        # Extract 1-D SHAP values for the sample
        if isinstance(shap_values, list):
            sv = shap_values[0][0].flatten()
        else:
            sv = shap_values[0].flatten()

        top_idx = np.argsort(np.abs(sv))[-10:][::-1]
        axes[1].barh([self.feature_names[i] for i in top_idx], sv[top_idx])
        axes[1].set_title("Top SHAP Feature Impacts")
        axes[1].set_xlabel("SHAP value")
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.show()
