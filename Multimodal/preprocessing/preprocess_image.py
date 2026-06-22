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


DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw/images")
PRE_IMG_DIR = os.path.join(DATA_DIR, "preprocessed/images")
os.makedirs(RAW_IMG_DIR, exist_ok=True)
os.makedirs(PRE_IMG_DIR, exist_ok=True)
import pandas as pd


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
    df_meta = pd.read_csv(os.path.join(DATA_DIR, "raw/metadata.csv"))
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

    hdf5_local = os.path.join(DATA_DIR, "raw/train-image.hdf5")
    print(f"\nSử dụng HDF5 tại {hdf5_local} ...")

    if not os.path.exists(hdf5_local):
        raise FileNotFoundError(f"Không tìm thấy {hdf5_local}. Hãy đảm bảo DVC đã pull dữ liệu.")

    print("Bắt đầu trích xuất + tiền xử lý ảnh...")
    extracted = preprocessed = skipped = errors = 0

    # Bulk check existing files to avoid recreating
    existing_raw = set(os.listdir(RAW_IMG_DIR)) if os.path.exists(RAW_IMG_DIR) else set()
    existing_pre = set(os.listdir(PRE_IMG_DIR)) if os.path.exists(PRE_IMG_DIR) else set()

    with h5py.File(hdf5_local, "r") as hf:
        all_keys = list(hf.keys())
        for isic_id in tqdm(all_keys, desc="Processing"):
            if isic_id not in selected_ids:
                continue

            try:
                img_bytes = bytes(hf[isic_id][()])

                # ── raw/images/<id>.jpg ← bytes gốc từ HDF5
                raw_filename = f"{isic_id}.jpg"
                if raw_filename not in existing_raw:
                    with open(os.path.join(RAW_IMG_DIR, raw_filename), "wb") as f:
                        f.write(img_bytes)
                    extracted += 1

                # ── preprocessed/images/<id>.png ← sau pipeline
                pre_filename = f"{isic_id}.png"
                if pre_filename not in existing_pre:
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_arr = np.array(img_pil)
                    img_proc = preprocess_image(img_arr)

                    Image.fromarray(img_proc).save(os.path.join(PRE_IMG_DIR, pre_filename), format="PNG")
                    preprocessed += 1
                else:
                    skipped += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Lỗi {isic_id}: {e}")
    print(f"\nKết quả:")
    print(f"  Đã extract : {extracted:,} ảnh → {RAW_IMG_DIR}")
    print(f"  Đã preprocess: {preprocessed:,} → {PRE_IMG_DIR}")
    print(f"  Bỏ qua (đã có): {skipped:,}")
    print(f"  Lỗi: {errors}")
    print("\nBước 2a hoàn thành!")


if __name__ == "__main__":
    main()
