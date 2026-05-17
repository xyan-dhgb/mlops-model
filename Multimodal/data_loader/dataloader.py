"""
dataloader.py — Bước 3: Tạo features array + stratified splits → S3

Đọc :
  s3://kltn-isic-2024-colab/preprocessed/metadata_clean.csv
  s3://kltn-isic-2024-colab/preprocessed/encoders.pkl
  s3://kltn-isic-2024-colab/preprocessed/images/<isic_id>.png  (streaming)

Ghi (khớp cấu trúc notebook cell 70):
  s3://kltn-isic-2024-colab/features/X_tabular.npy
  s3://kltn-isic-2024-colab/features/X_images.npy
  s3://kltn-isic-2024-colab/features/y_labels.npy

  s3://kltn-isic-2024-colab/splits/train/X_tab_train.npy
  s3://kltn-isic-2024-colab/splits/train/X_img_train.npy
  s3://kltn-isic-2024-colab/splits/train/y_train.npy
  s3://kltn-isic-2024-colab/splits/val/X_tab_val.npy
  s3://kltn-isic-2024-colab/splits/val/X_img_val.npy
  s3://kltn-isic-2024-colab/splits/val/y_val.npy
  s3://kltn-isic-2024-colab/splits/test/X_tab_test.npy
  s3://kltn-isic-2024-colab/splits/test/X_img_test.npy
  s3://kltn-isic-2024-colab/splits/test/y_test.npy
  s3://kltn-isic-2024-colab/splits/split_info.json
"""
import io
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from s3_utils import (
    load_csv, load_pkl, download_bytes,
    save_npy, upload_bytes,
    list_s3_keys, s3_key_exists,
    S3_OUTPUT_BUCKET,
)

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
IMAGE_SIZE  = int(os.environ.get("IMAGE_SIZE", "224"))


def load_image_from_s3(isic_id: str) -> np.ndarray | None:
    """Tải ảnh đã preprocessed từ S3, trả về float32 [H,W,3] / None nếu lỗi."""
    key = f"preprocessed/images/{isic_id}.png"
    try:
        data = download_bytes(key, bucket=S3_OUTPUT_BUCKET)
        img  = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("BƯỚC 3: Tạo features array + splits → S3")
    print(f"  Bucket: s3://{S3_OUTPUT_BUCKET}/")
    print("=" * 60)

    if s3_key_exists("splits/split_info.json", bucket=S3_OUTPUT_BUCKET):
        print("\n✅ Tìm thấy 'splits/split_info.json' trên S3.")
        print("Dataloader đã chạy thành công trước đó. BỎ QUA (SKIPPED).")
        return

    df       = load_csv("preprocessed/metadata_clean.csv", bucket=S3_OUTPUT_BUCKET)
    encoders = load_pkl("preprocessed/encoders.pkl",      bucket=S3_OUTPUT_BUCKET)
    feature_cols = encoders["feature_cols"]

    # Kiểm tra ảnh có trên S3 không
    print("\nKiểm tra ảnh đã preprocessed trên S3...")
    existing_keys = list_s3_keys("preprocessed/images/", bucket=S3_OUTPUT_BUCKET)
    existing_ids  = {k.split("/")[-1].replace(".png", "") for k in existing_keys}
    df_available  = df[df["isic_id"].isin(existing_ids)].reset_index(drop=True)
    print(f"  Ảnh có sẵn: {len(existing_ids):,}")
    print(f"  Mẫu khớp  : {len(df_available):,}")

    # ── Build arrays (khớp notebook cell 32 + 39) ───────────────────────
    n_samples = len(df_available)
    print(f"\nĐang tải ảnh + build arrays (pre-allocated cho {n_samples} mẫu)...")

    # Pre-allocate arrays to avoid OOM memory spikes
    X_tabular = np.zeros((n_samples, len(feature_cols)), dtype=np.float32)
    X_images  = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    y_labels  = np.zeros(n_samples, dtype=np.int32)

    valid_count = 0
    for _, row in tqdm(df_available.iterrows(), total=n_samples):
        img = load_image_from_s3(row["isic_id"])
        if img is None:
            continue

        X_tabular[valid_count] = row[feature_cols].values.astype(np.float32)
        X_images[valid_count] = img
        y_labels[valid_count] = int(row["target"])
        valid_count += 1

    # Trim arrays if some images failed to load
    X_tabular = X_tabular[:valid_count]
    X_images  = X_images[:valid_count]
    y_labels  = y_labels[:valid_count]

    print(f"\nArrays built:")
    print(f"  X_tabular : {X_tabular.shape}")
    print(f"  X_images  : {X_images.shape}")
    print(f"  y_labels  : {y_labels.shape} (Mal={y_labels.sum()}, Ben={(y_labels==0).sum()})")

    # Lưu features (khớp cell 39 của notebook)
    save_npy(X_tabular, "features/X_tabular.npy", bucket=S3_OUTPUT_BUCKET)
    save_npy(X_images,  "features/X_images.npy",  bucket=S3_OUTPUT_BUCKET)
    save_npy(y_labels,  "features/y_labels.npy",  bucket=S3_OUTPUT_BUCKET)

    # ── Stratified split 64/16/20 (khớp cell 70) ────────────────────────
    idx = np.arange(len(y_labels))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.20, stratify=y_labels, random_state=RANDOM_SEED
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.20,
        stratify=y_labels[idx_trainval], random_state=RANDOM_SEED
    )

    splits = {
        "train": (idx_train, "splits/train/"),
        "val":   (idx_val,   "splits/val/"),
        "test":  (idx_test,  "splits/test/"),
    }

    # Lưu splits (khớp chính xác tên file notebook cell 70)
    split_info = {"random_seed": RANDOM_SEED, "splits": {}}
    for name, (idx_s, prefix) in splits.items():
        xs = X_tabular[idx_s]
        xi = X_images[idx_s]
        ys = y_labels[idx_s]

        save_npy(xs, f"{prefix}X_tab_{name}.npy",  bucket=S3_OUTPUT_BUCKET)
        save_npy(xi, f"{prefix}X_img_{name}.npy",  bucket=S3_OUTPUT_BUCKET)
        save_npy(ys, f"{prefix}y_{name}.npy",       bucket=S3_OUTPUT_BUCKET)

        split_info["splits"][name] = {
            "total":     len(ys),
            "malignant": int(ys.sum()),
            "benign":    int((ys == 0).sum()),
            "ratio_mal": round(float(ys.mean()), 4),
        }
        print(f"  {name:5s}: {len(ys):>6,} mẫu | "
              f"Mal={ys.sum()} ({100*ys.mean():.1f}%)")

    upload_bytes(
        json.dumps(split_info, indent=2).encode(),
        "splits/split_info.json",
        bucket=S3_OUTPUT_BUCKET,
    )
    print(f"\nSplit info → s3://{S3_OUTPUT_BUCKET}/splits/split_info.json")
    print("\nBước 3 hoàn thành!")


if __name__ == "__main__":
    main()
