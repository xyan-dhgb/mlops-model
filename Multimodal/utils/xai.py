"""
xai.py — Bước 7: Explainable AI (Grad-CAM + SHAP DeepExplainer)
  - Grad-CAM  : Heatmap vùng ảnh ảnh hưởng quyết định (Image Branch)
  - SHAP      : Đóng góp từng feature tabular (Tabular Branch)
Đầu vào:
  /data/output/best_model_isic2024.h5
  /data/processed/{tabular_processed.pkl, encoders.pkl}
  /data/processed/images/<isic_id>.png
  /data/splits/test_idx.npy
  /data/eval/best_threshold.txt
Đầu ra:
  /data/xai/gradcam/gradcam_<isic_id>.png
  /data/xai/shap_values.npy
  /data/xai/shap_waterfall_<i>.png
  /data/xai/shap_global_bar.png
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

PROCESSED_DIR     = os.environ.get("PROCESSED_DIR", "/data/processed")
SPLITS_DIR        = os.environ.get("SPLITS_DIR", "/data/splits")
OUTPUT_DIR        = os.environ.get("OUTPUT_DIR", "/data/output")
EVAL_DIR          = os.environ.get("EVAL_DIR", "/data/eval")
XAI_DIR           = os.environ.get("XAI_DIR", "/data/xai")
NUM_GRADCAM       = int(os.environ.get("NUM_GRADCAM_SAMPLES", "20"))
NUM_SHAP          = int(os.environ.get("NUM_SHAP_SAMPLES", "100"))
SHAP_BACKGROUND   = int(os.environ.get("SHAP_BACKGROUND", "50"))
IMAGE_SIZE        = int(os.environ.get("IMAGE_SIZE", "224"))

GRADCAM_DIR = os.path.join(XAI_DIR, "gradcam")
os.makedirs(XAI_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

IMAGE_DIR  = os.path.join(PROCESSED_DIR, "images")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_isic2024.h5")
CONV_LAST  = "top_conv"   # Layer conv cuối của EfficientNetB3


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    focal_loss_fn.__name__ = "focal_loss"
    return focal_loss_fn


# ── Grad-CAM ─────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array: np.ndarray,
                     tab_array: np.ndarray,
                     conv_layer_name: str = CONV_LAST) -> np.ndarray:
    """
    Tính Grad-CAM heatmap cho một ảnh.
    Returns: heatmap numpy array [H, W] chuẩn hóa 0–1
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output],
    )
    img_batch = img_array[np.newaxis]
    tab_batch = tab_array[np.newaxis]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            {"image_input": img_batch, "tabular_input": tab_batch}
        )
        loss = predictions[:, 0]

    grads   = tape.gradient(loss, conv_outputs)[0]          # [H, W, C]
    weights = tf.reduce_mean(grads, axis=(0, 1))            # Global Average Pooling
    cam     = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    cam     = tf.nn.relu(cam).numpy()                       # ReLU: giữ vùng dương
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_heatmap(orig_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Chồng heatmap (colormap jet) lên ảnh gốc."""
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (orig_img.shape[1], orig_img.shape[0]), Image.BILINEAR
        )
    )
    colormap = plt.cm.jet(heatmap_resized / 255.0)[:, :, :3]
    overlay  = (alpha * colormap + (1 - alpha) * orig_img / 255.0)
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def save_gradcam(isic_id: str, orig_img: np.ndarray,
                  heatmap: np.ndarray, prob: float, label: int):
    overlay = overlay_heatmap(orig_img, heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(orig_img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM | p={prob:.3f} | "
                      f"{'Malignant' if label==1 else 'Benign'}")
    axes[1].axis("off")
    plt.suptitle(f"ISIC ID: {isic_id}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRADCAM_DIR, f"gradcam_{isic_id}.png"), dpi=120)
    plt.close()


# ── SHAP ─────────────────────────────────────────────────────────────────
def run_shap(model, X_tab_bg: np.ndarray, X_img_bg: np.ndarray,
              X_tab_test: np.ndarray, X_img_test: np.ndarray,
              feature_cols: list):
    import shap

    # Sub-model: chỉ nhận tabular → output (giữ image cố định)
    @tf.function
    def tabular_predict(tab_input):
        img_fixed = tf.zeros([tf.shape(tab_input)[0], IMAGE_SIZE, IMAGE_SIZE, 3],
                              dtype=tf.float32)
        return model({"image_input": img_fixed, "tabular_input": tab_input})

    explainer    = shap.DeepExplainer(tabular_predict, X_tab_bg)
    shap_values  = explainer.shap_values(X_tab_test)
    shap_values  = np.array(shap_values).squeeze()

    # Lưu SHAP values
    np.save(os.path.join(XAI_DIR, "shap_values.npy"), shap_values)

    # Waterfall plots (mỗi mẫu)
    for i in range(min(5, len(X_tab_test))):
        exp = shap.Explanation(
            values=shap_values[i],
            base_values=float(explainer.expected_value),
            data=X_tab_test[i],
            feature_names=feature_cols,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f"shap_waterfall_{i}.png"), dpi=120)
        plt.close()

    # Global bar plot (Top 20 features)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[-20:][::-1]
    top20_names = [feature_cols[i] for i in top20_idx]
    top20_vals  = mean_abs[top20_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(20), top20_vals[::-1], color="steelblue")
    plt.yticks(range(20), top20_names[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP Global Feature Importance (Top 20 Tabular Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "shap_global_bar.png"), dpi=150)
    plt.close()
    print(f"SHAP hoàn thành. shap_values.npy và plots → {XAI_DIR}")


def main():
    print("Đang tải model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"focal_loss_fn": focal_loss()},
    )

    df       = pd.read_pickle(os.path.join(PROCESSED_DIR, "tabular_processed.pkl"))
    encoders = pickle.load(open(os.path.join(PROCESSED_DIR, "encoders.pkl"), "rb"))
    feature_cols = encoders["feature_cols"]

    idx_test = np.load(os.path.join(SPLITS_DIR, "test_idx.npy"))
    best_thr_path = os.path.join(EVAL_DIR, "best_threshold.txt")
    best_thr = float(open(best_thr_path).read().strip()) if os.path.exists(best_thr_path) else 0.5

    records = df.iloc[idx_test].reset_index(drop=True)

    # ── Grad-CAM trên NUM_GRADCAM mẫu ───────────────────────────────────
    print(f"\nGrad-CAM trên {NUM_GRADCAM} mẫu...")
    gradcam_imgs, gradcam_tabs = [], []

    for i, (_, row) in enumerate(tqdm(records.iterrows(), total=len(records))):
        if i >= NUM_GRADCAM * 3:
            break
        img_path = os.path.join(IMAGE_DIR, f"{row['isic_id']}.png")
        if not os.path.exists(img_path):
            continue

        orig_img = np.array(Image.open(img_path).convert("RGB"))
        img_arr  = orig_img.astype(np.float32) / 255.0
        tab_arr  = row[feature_cols].values.astype(np.float32)

        prob     = float(model.predict(
            {"image_input": img_arr[np.newaxis],
             "tabular_input": tab_arr[np.newaxis]},
            verbose=0,
        )[0, 0])

        heatmap = compute_gradcam(model, img_arr, tab_arr)
        save_gradcam(row["isic_id"], orig_img, heatmap,
                     prob, int(row["target"]))

        if len(gradcam_imgs) < NUM_GRADCAM:
            gradcam_imgs.append(img_arr)
            gradcam_tabs.append(tab_arr)

        if len(gradcam_imgs) >= NUM_GRADCAM:
            break

    print(f"Grad-CAM plots → {GRADCAM_DIR}")

    # ── SHAP ─────────────────────────────────────────────────────────────
    print(f"\nSHAP DeepExplainer trên {NUM_SHAP} mẫu (background={SHAP_BACKGROUND})...")
    all_imgs, all_tabs = [], []
    for _, row in records.iterrows():
        img_path = os.path.join(IMAGE_DIR, f"{row['isic_id']}.png")
        if not os.path.exists(img_path):
            continue
        all_imgs.append(
            np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        all_tabs.append(row[feature_cols].values.astype(np.float32))
        if len(all_imgs) >= NUM_SHAP + SHAP_BACKGROUND:
            break

    all_imgs = np.array(all_imgs)
    all_tabs = np.array(all_tabs)

    X_img_bg   = all_imgs[:SHAP_BACKGROUND]
    X_tab_bg   = all_tabs[:SHAP_BACKGROUND]
    X_img_test = all_imgs[SHAP_BACKGROUND:SHAP_BACKGROUND + NUM_SHAP]
    X_tab_test = all_tabs[SHAP_BACKGROUND:SHAP_BACKGROUND + NUM_SHAP]

    run_shap(model, X_tab_bg, X_img_bg, X_tab_test, X_img_test, feature_cols)
    print("\nXAI hoàn thành!")


if __name__ == "__main__":
    main()
