"""
xai.py — Bước 7: Grad-CAM + SHAP DeepExplainer

Đọc từ S3:
  preprocessed/best_model_isic2024.h5
  preprocessed/encoders.pkl
  preprocessed/best_threshold.txt
  preprocessed/images/<isic_id>.png   ← ảnh đã preprocessed
  splits/test/X_tab_test.npy, X_img_test.npy, y_test.npy

Ghi lên S3:
  preprocessed/xai/gradcam/<isic_id>.png
  preprocessed/xai/shap_values.npy
  preprocessed/xai/shap_waterfall_<i>.png
  preprocessed/xai/shap_global_bar.png
"""
import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from s3_utils import (
    load_npy, load_pkl, load_keras_model,
    download_bytes, upload_bytes, save_npy,
    list_s3_keys,
    S3_OUTPUT_BUCKET,
)

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

    s3_key = f"{prefix}{isic_id}.png"
    upload_bytes(fig_to_bytes(fig), s3_key, bucket=S3_OUTPUT_BUCKET)
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

    save_npy(shap_vals, f"{prefix}shap_values.npy", bucket=S3_OUTPUT_BUCKET)

    # Waterfall plots (5 mẫu đầu)
    expected_val = float(np.array(explainer.expected_value).ravel()[0])
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
        key = f"{prefix}shap_waterfall_{i}.png"
        upload_bytes(fig_to_bytes(plt.gcf()), key, bucket=S3_OUTPUT_BUCKET)
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
    upload_bytes(fig_to_bytes(fig), f"{prefix}shap_global_bar.png",
                 bucket=S3_OUTPUT_BUCKET)
    plt.close()
    print(f"SHAP → s3://{S3_OUTPUT_BUCKET}/{prefix}")


def main():
    print("=" * 60)
    print("BƯỚC 7: XAI — Grad-CAM + SHAP")
    print(f"  Bucket: s3://{S3_OUTPUT_BUCKET}/preprocessed/xai/")
    print("=" * 60)

    GRADCAM_PREFIX = "preprocessed/xai/gradcam/"
    XAI_PREFIX     = "preprocessed/xai/"

    model = load_keras_model(
        "preprocessed/best_model_isic2024.h5",
        bucket=S3_OUTPUT_BUCKET,
        custom_objects={"focal_loss": focal_loss()},
    )
    encoders     = load_pkl("preprocessed/encoders.pkl", bucket=S3_OUTPUT_BUCKET)
    feature_cols = encoders["feature_cols"]

    best_thr = float(
        download_bytes("preprocessed/best_threshold.txt",
                       bucket=S3_OUTPUT_BUCKET).decode().strip()
    )

    X_tab_test = load_npy("splits/test/X_tab_test.npy", bucket=S3_OUTPUT_BUCKET)
    X_img_test = load_npy("splits/test/X_img_test.npy", bucket=S3_OUTPUT_BUCKET)
    y_test     = load_npy("splits/test/y_test.npy",     bucket=S3_OUTPUT_BUCKET)

    # ── Grad-CAM ─────────────────────────────────────────────────────────
    print(f"\nGrad-CAM trên {NUM_GRADCAM} mẫu...")
    # Lấy isic_id từ preprocessed/images/
    image_keys = list_s3_keys("preprocessed/images/", bucket=S3_OUTPUT_BUCKET)
    isic_ids   = [k.split("/")[-1].replace(".png", "") for k in image_keys][:NUM_GRADCAM * 3]

    done_gradcam = 0
    for isic_id in isic_ids:
        if done_gradcam >= NUM_GRADCAM:
            break
        try:
            img_data = download_bytes(f"preprocessed/images/{isic_id}.png",
                                       bucket=S3_OUTPUT_BUCKET)
            img_float = np.array(
                Image.open(io.BytesIO(img_data)).convert("RGB"), dtype=np.float32
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

    print(f"Grad-CAM → s3://{S3_OUTPUT_BUCKET}/{GRADCAM_PREFIX}")

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

    print("\nBước 7 hoàn thành!")


if __name__ == "__main__":
    main()
