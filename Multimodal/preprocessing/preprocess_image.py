"""
preprocess_image.py — Bước 2a: Tiền xử lý ảnh da liễu từ HDF5
Pipeline: Resize → CLAHE → Gaussian Blur → Contrast ×1.2 → Lưu PNG
Đầu vào : /data/raw/train-image.hdf5
Đầu ra  : /data/processed/images/<isic_id>.png
"""
import os
import io
import numpy as np
import cv2
import h5py
from PIL import Image, ImageEnhance
from tqdm import tqdm

RAW_DIR       = os.environ.get("RAW_DIR", "/data/raw")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/data/processed")
IMAGE_SIZE    = int(os.environ.get("IMAGE_SIZE", "224"))
MAX_IMAGES    = os.environ.get("MAX_IMAGES", "")  # "" = xử lý tất cả

HDF5_PATH  = os.path.join(RAW_DIR, "train-image.hdf5")
OUTPUT_DIR = os.path.join(PROCESSED_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_image(img_array: np.ndarray,
                     apply_clahe: bool = True,
                     apply_gaussian: bool = True,
                     enhance_contrast: float = 1.2) -> np.ndarray:
    """
    Áp dụng pipeline tiền xử lý ảnh da liễu:
      1. Resize về IMAGE_SIZE × IMAGE_SIZE
      2. CLAHE (clipLimit=2.0, tileGridSize=8×8) — cân bằng histogram cục bộ
      3. Gaussian Blur (kernel 3×3) — giảm nhiễu
      4. Tăng contrast 20%
    """
    # Resize
    img = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE),
                     interpolation=cv2.INTER_AREA)

    # CLAHE trên kênh L của không gian LAB
    if apply_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Gaussian Blur
    if apply_gaussian:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)

    # Tăng contrast ×1.2
    if enhance_contrast != 1.0:
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        img = np.array(enhancer.enhance(enhance_contrast))

    return img


def main():
    print(f"Mở HDF5: {HDF5_PATH}")
    with h5py.File(HDF5_PATH, "r") as f:
        keys = list(f.keys())
        if MAX_IMAGES:
            keys = keys[:int(MAX_IMAGES)]
        print(f"Số ảnh cần xử lý: {len(keys)}")

        for isic_id in tqdm(keys, desc="Preprocessing images"):
            out_path = os.path.join(OUTPUT_DIR, f"{isic_id}.png")
            if os.path.exists(out_path):
                continue  # Bỏ qua nếu đã xử lý

            # Đọc bytes từ HDF5
            img_bytes = np.array(f[isic_id])
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_arr = np.array(img_pil)

            img_processed = preprocess_image(img_arr)

            Image.fromarray(img_processed).save(out_path, format="PNG")

    print(f"Hoàn thành! Ảnh đã lưu vào {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
