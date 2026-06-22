"""
xai.py — Bước 7: Grad-CAM + SHAP DeepExplainer

Đọc từ Local/DVC:
  Local/DVC (final/best_model_isic2024.h5)
  Local/DVC (preprocessed/encoders.pkl)
  Local/DVC (final/best_threshold.txt)
  Local/DVC (preprocessed/images/<isic_id>.png)
  Local/DVC (splits/test/X_tab_test.npy, X_img_test.npy, y_test.npy)
# Cũ: s3://kltn-isic-2024-colab/preprocessed/best_model_isic2024.h5 ...

Ghi lên Local/DVC:
  Local/DVC (final/xai/gradcam/<isic_id>.png)
  Local/DVC (final/xai/shap_values.npy)
  Local/DVC (final/xai/shap_waterfall_<i>.png)
  Local/DVC (final/xai/shap_global_bar.png)
# Cũ: s3://kltn-isic-2024-colab/preprocessed/xai/...
"""
import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import concurrent.futures
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import s3_utils
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import pickle
from tensorflow.keras.models import load_model

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
os.makedirs(os.path.join(DATA_DIR, "final/xai/gradcam"), exist_ok=True)


NUM_GRADCAM = int(os.environ.get("NUM_GRADCAM_SAMPLES", "20"))
NUM_SHAP    = int(os.environ.get("NUM_SHAP_SAMPLES", "100"))
SHAP_BG     = int(os.environ.get("SHAP_BACKGROUND", "50"))
IMAGE_SIZE  = int(os.environ.get("IMAGE_SIZE", "224"))
CONV_LAST   = "top_conv"   # conv cuối EfficientNetB3


def focal_loss(gamma=2.0, alpha=0.25):
    def fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    fn.__name__ = "focal_loss"
    return fn


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ── Grad-CAM ─────────────────────────────────────────────────────────────
def compute_gradcam(model, img_arr, tab_arr):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(CONV_LAST).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model({
            "image_input":   tf.constant(img_arr[np.newaxis]),
            "tabular_input": tf.constant(tab_arr[np.newaxis]),
        })
        loss = pred[:, 0]
    grads   = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam     = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam     = tf.nn.relu(cam).numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_and_save(model, isic_id, img_float, tab_arr, label, prob, prefix):
    cam     = compute_gradcam(model, img_float, tab_arr)
    orig    = (img_float * 255).astype(np.uint8)
    heat    = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                  (orig.shape[1], orig.shape[0]), Image.BILINEAR))
    color   = plt.cm.jet(heat / 255.0)[:, :, :3]
    overlay = np.clip(0.45 * color + 0.55 * orig / 255.0, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(orig); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM | p={prob:.3f} | {'Malignant' if label==1 else 'Benign'}")
    axes[1].axis("off")
    plt.suptitle(f"ISIC ID: {isic_id}", fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(DATA_DIR, f"{prefix}{isic_id}.png")
    with open(out_path, "wb") as f:
        f.write(fig_to_bytes(fig))
    plt.close()


# ── SHAP ─────────────────────────────────────────────────────────────────
def run_shap(model, X_tab_bg, X_img_bg, X_tab_test, X_img_test, feature_cols, prefix):
    import shap

    # GradientExplainer nhận model TF trực tiếp — không cần @tf.function wrapper.
    # Inputs phải là list theo đúng thứ tự model.inputs: [image_input, tabular_input]
    background = [X_img_bg,   X_tab_bg]    # (SHAP_BG, H, W, 3), (SHAP_BG, tabular_dim)
    test_data  = [X_img_test, X_tab_test]  # (NUM_SHAP, ...)

    explainer     = shap.GradientExplainer(model, background)
    shap_vals_all = explainer.shap_values(test_data)
    # shap_vals_all → list[array] — 1 array per model input
    # index 0 = image shap, index 1 = tabular shap (shape: n × tabular_dim)
    shap_vals = np.array(shap_vals_all[1]).squeeze()

    np.save(os.path.join(DATA_DIR, f"{prefix}shap_values.npy"), shap_vals)

    # Waterfall plots (5 mẫu đầu)
    # GradientExplainer không có .expected_value — tự tính bằng mean prediction
    # trên background dataset (đây chính là định nghĩa của expected value trong SHAP)
    bg_preds = model.predict(
        {"image_input": X_img_bg, "tabular_input": X_tab_bg}, verbose=0
    )
    expected_val = float(bg_preds.mean())
    for i in range(min(5, len(X_tab_test))):
        exp = shap.Explanation(
            values=shap_vals[i],
            base_values=expected_val,
            data=X_tab_test[i],
            feature_names=feature_cols,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        key = os.path.join(DATA_DIR, f"{prefix}shap_waterfall_{i}.png")
        with open(key, "wb") as f:
            f.write(fig_to_bytes(plt.gcf()))
        plt.close()

    # Global bar (Top 20)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top20    = np.argsort(mean_abs)[-20:][::-1]
    fig, ax  = plt.subplots(figsize=(10, 6))
    ax.barh(range(20), mean_abs[top20][::-1], color="steelblue")
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_cols[i] for i in top20][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Global Feature Importance (Top 20 Tabular)")
    plt.tight_layout()
    with open(os.path.join(DATA_DIR, f"{prefix}shap_global_bar.png"), "wb") as f:
        f.write(fig_to_bytes(fig))
    plt.close()
    print(f"SHAP → {DATA_DIR}/{prefix}")


def main():
    print("=" * 60)
    print("BƯỚC 7: XAI — Grad-CAM + SHAP")
    print(f"  Bucket: {DATA_DIR}/final/xai/")
    print("=" * 60)

    GRADCAM_PREFIX = "final/xai/gradcam/"
    XAI_PREFIX     = "final/xai/"

    model = load_model(
        os.path.join(DATA_DIR, "final/best_model_isic2024.h5"),
        compile=False,
        custom_objects={"focal_loss": focal_loss()},
    )
    with open(os.path.join(DATA_DIR, "preprocessed/encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    feature_cols = encoders["feature_cols"]

    with open(os.path.join(DATA_DIR, "final/best_threshold.txt"), "r") as f:
        best_thr = float(f.read().strip())

    X_tab_test = np.load(os.path.join(DATA_DIR, "splits/test/X_tab_test.npy"))
    X_img_test = np.load(os.path.join(DATA_DIR, "splits/test/X_img_test.npy"), mmap_mode="r")
    y_test     = np.load(os.path.join(DATA_DIR, "splits/test/y_test.npy"))

    # ── Grad-CAM ─────────────────────────────────────────────────────────
    print(f"\nGrad-CAM trên {NUM_GRADCAM} mẫu...")
    # Lấy isic_id từ local preprocessed/images/
    img_dir = os.path.join(DATA_DIR, "preprocessed/images")
    image_keys = os.listdir(img_dir) if os.path.exists(img_dir) else []
    isic_ids   = [k.replace(".png", "") for k in image_keys][:NUM_GRADCAM * 3]

    done_gradcam = 0
    for isic_id in isic_ids:
        if done_gradcam >= NUM_GRADCAM:
            break
        try:
            img_path = os.path.join(DATA_DIR, f"preprocessed/images/{isic_id}.png")
            img_float = np.array(
                Image.open(img_path).convert("RGB"), dtype=np.float32
            ) / 255.0
        except Exception:
            continue

        # Tìm tab_arr tương ứng (dùng index nếu có)
        if done_gradcam >= len(X_tab_test):
            break
        tab_arr = X_tab_test[done_gradcam]
        label   = int(y_test[done_gradcam])

        prob = float(model.predict(
            {"image_input":   img_float[np.newaxis],
             "tabular_input": tab_arr[np.newaxis]},
            verbose=0)[0, 0])

        overlay_and_save(model, isic_id, img_float, tab_arr,
                         label, prob, GRADCAM_PREFIX)
        done_gradcam += 1

    print(f"Grad-CAM → {DATA_DIR}/{GRADCAM_PREFIX}")

    # ── SHAP ─────────────────────────────────────────────────────────────
    print(f"\nSHAP trên {NUM_SHAP} mẫu (background={SHAP_BG})...")
    total = min(SHAP_BG + NUM_SHAP, len(X_tab_test))
    run_shap(
        model,
        X_tab_bg   = X_tab_test[:SHAP_BG].astype(np.float32),
        X_img_bg   = X_img_test[:SHAP_BG].astype(np.float32),
        X_tab_test = X_tab_test[SHAP_BG:total].astype(np.float32),
        X_img_test = X_img_test[SHAP_BG:total].astype(np.float32),
        feature_cols = feature_cols,
        prefix       = XAI_PREFIX,
    )
    print("\nĐồng bộ thư mục lên S3 Output Bucket (thay thế DVC)...")

    def upload_worker(local_path):
        rel_path = os.path.relpath(local_path, DATA_DIR)
        s3_key = rel_path.replace("\\", "/")
        s3_utils.upload_file(local_path, s3_key)
        
    upload_list = []
    xai_dir = os.path.join(DATA_DIR, XAI_PREFIX)
    for root, dirs, files in os.walk(xai_dir):
        for f in files:
            upload_list.append(os.path.join(root, f))
            
    print(f"Bắt đầu upload {len(upload_list)} file XAI bằng đa luồng...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(upload_worker, upload_list), total=len(upload_list), desc="Uploading"))

    print("\nBước 7 hoàn thành!")


if __name__ == "__main__":
    main()
