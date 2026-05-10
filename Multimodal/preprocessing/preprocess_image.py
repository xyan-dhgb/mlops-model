"""
preprocess_image.py — Bước 2a: Trích xuất HDF5 → tiền xử lý → lưu S3

Đọc  : s3://kltn-isic-2024-challenge/isic-2024-challenge/train-image.hdf5  (stream)
        s3://kltn-isic-2024-colab/raw/metadata.csv  (nhãn + isic_id)

Ghi  :
  s3://kltn-isic-2024-colab/raw/images/<isic_id>.jpg          ← bytes gốc từ HDF5
  s3://kltn-isic-2024-colab/preprocessed/images/<isic_id>.png ← sau CLAHE+Gaussian+Contrast

Pipeline ảnh (khớp notebook cell 28):
  Resize(224×224) → CLAHE(clipLimit=2.0, tile=8×8) → GaussianBlur(3×3) → Contrast×1.2
"""
import io
import os
import tempfile
import numpy as np
import cv2
import h5py
from PIL import Image, ImageEnhance
from tqdm import tqdm

from s3_utils import (
    get_s3_client,
    download_bytes, upload_bytes, load_csv,
    s3_key_exists,
    S3_INPUT_BUCKET, S3_INPUT_PREFIX, S3_OUTPUT_BUCKET,
)

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
MAX_IMAGES = os.environ.get("MAX_IMAGES", "")    # "" = tất cả


def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """
    Pipeline tiền xử lý khớp notebook cell 28:
      1. Resize → IMAGE_SIZE × IMAGE_SIZE
      2. CLAHE (clipLimit=2.0, tileGridSize=8×8) trên kênh L của LAB
      3. GaussianBlur kernel (3,3)
      4. Contrast ×1.2
    Trả về: uint8 RGB [H, W, 3]
    """
    img = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE),
                     interpolation=cv2.INTER_AREA)

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Contrast ×1.2
    pil = Image.fromarray(img)
    img = np.array(ImageEnhance.Contrast(pil).enhance(1.2))

    return img.astype(np.uint8)


def main():
    print("=" * 60)
    print("BƯỚC 2a: Tiền xử lý ảnh từ HDF5 → S3")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("=" * 60)

    # Đọc metadata để biết danh sách isic_id + nhãn
    df_meta = load_csv("raw/metadata.csv", bucket=S3_OUTPUT_BUCKET)
    print(f"Metadata: {len(df_meta):,} mẫu")

    mal_ids = df_meta[df_meta["target"] == 1]["isic_id"].tolist()
    ben_ids = df_meta[df_meta["target"] == 0]["isic_id"].tolist()

    # Lấy TẤT CẢ Malignant + sample Benign (khớp notebook cell 25)
    n_mal = len(mal_ids)
    n_ben = min(10000, len(ben_ids))
    np.random.seed(42)
    selected_ben = np.random.choice(ben_ids, size=n_ben, replace=False).tolist()
    selected_ids = set(mal_ids + selected_ben)

    if MAX_IMAGES:
        selected_ids = set(list(selected_ids)[:int(MAX_IMAGES)])

    print(f"Sẽ xử lý: {len(selected_ids):,} ảnh "
          f"(Mal={n_mal}, Ben={n_ben})")

    # Tải HDF5 vào file tạm (streaming từ S3)
    hdf5_key = f"{S3_INPUT_PREFIX}/train-image.hdf5"
    print(f"\nĐang tải HDF5 từ s3://{S3_INPUT_BUCKET}/{hdf5_key} ...")
    s3 = get_s3_client()
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        s3.download_file(S3_INPUT_BUCKET, hdf5_key, tmp.name)
        hdf5_local = tmp.name

    print("Bắt đầu trích xuất + tiền xử lý ảnh...")
    extracted = preprocessed = skipped = errors = 0

    with h5py.File(hdf5_local, "r") as hf:
        all_keys = list(hf.keys())
        for isic_id in tqdm(all_keys, desc="Processing"):
            if isic_id not in selected_ids:
                continue

            try:
                img_bytes = bytes(hf[isic_id][()])

                # ── raw/images/<id>.jpg ← bytes gốc từ HDF5
                raw_key = f"raw/images/{isic_id}.jpg"
                if not s3_key_exists(raw_key, S3_OUTPUT_BUCKET):
                    upload_bytes(img_bytes, raw_key, S3_OUTPUT_BUCKET)
                extracted += 1

                # ── preprocessed/images/<id>.png ← sau pipeline
                pre_key = f"preprocessed/images/{isic_id}.png"
                if not s3_key_exists(pre_key, S3_OUTPUT_BUCKET):
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_arr = np.array(img_pil)
                    img_proc = preprocess_image(img_arr)

                    buf = io.BytesIO()
                    Image.fromarray(img_proc).save(buf, format="PNG")
                    upload_bytes(buf.getvalue(), pre_key, S3_OUTPUT_BUCKET)
                    preprocessed += 1
                else:
                    skipped += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Lỗi {isic_id}: {e}")

    os.unlink(hdf5_local)
    print(f"\nKết quả:")
    print(f"  Đã extract : {extracted:,} ảnh → s3://{S3_OUTPUT_BUCKET}/raw/images/")
    print(f"  Đã preprocess: {preprocessed:,} → s3://{S3_OUTPUT_BUCKET}/preprocessed/images/")
    print(f"  Bỏ qua (đã có): {skipped:,}")
    print(f"  Lỗi: {errors}")
    print("\nBước 2a hoàn thành!")


if __name__ == "__main__":
    main()
