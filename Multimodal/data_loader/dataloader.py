"""
dataloader.py — Bước 3: Tạo features array + stratified splits + xử lý mất cân bằng → Local/DVC

Đọc :
  Local/DVC (preprocessed/metadata_clean.csv)
  Local/DVC (preprocessed/encoders.pkl)
  Local/DVC (preprocessed/images/<isic_id>.png)
# Cũ: s3://kltn-isic-2024-colab/preprocessed/metadata_clean.csv
# Cũ: s3://kltn-isic-2024-colab/preprocessed/encoders.pkl
# Cũ: s3://kltn-isic-2024-colab/preprocessed/images/<isic_id>.png  (streaming)

Ghi (khớp cấu trúc notebook):
  Local/DVC (features/X_tabular.npy)
  Local/DVC (features/X_images.npy)
  Local/DVC (features/y_labels.npy)

  Local/DVC (splits/train/X_tab_train.npy)
  Local/DVC (splits/train/X_img_train.npy)
  Local/DVC (splits/train/y_train.npy)
  Local/DVC (splits/train/X_tab_train_os.npy)   ← sau oversampling
  Local/DVC (splits/train/X_img_train_os.npy)   ← sau oversampling
  Local/DVC (splits/train/y_train_os.npy)       ← sau oversampling
  Local/DVC (splits/val/X_tab_val.npy)
  Local/DVC (splits/val/X_img_val.npy)
  Local/DVC (splits/val/y_val.npy)
  Local/DVC (splits/test/X_tab_test.npy)
  Local/DVC (splits/test/X_img_test.npy)
  Local/DVC (splits/test/y_test.npy)
  Local/DVC (splits/split_info.json)
# Cũ: s3://kltn-isic-2024-colab/features/... và splits/...
"""
import io
import json
import os
import sys
import pickle
import concurrent.futures

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import s3_utils
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
os.makedirs(os.path.join(DATA_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "splits/train"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "splits/val"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "splits/test"), exist_ok=True)


RANDOM_SEED      = int(os.environ.get("RANDOM_SEED", "42"))
IMAGE_SIZE       = int(os.environ.get("IMAGE_SIZE", "224"))
OVERSAMPLE_RATIO = float(os.environ.get("OVERSAMPLE_RATIO", "0.25"))   # target Malignant ratio


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(isic_id: str) -> np.ndarray | None:
    """Tải ảnh đã preprocessed từ local, trả về float32 [H,W,3] / None nếu lỗi."""
    path = os.path.join(DATA_DIR, f"preprocessed/images/{isic_id}.png")
    try:
        img  = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_image(img_array: np.ndarray,
                  rotation_range: float = 15,
                  brightness_range: tuple = (0.8, 1.2),
                  zoom_range: float = 0.1,
                  strong: bool = False) -> np.ndarray:
    """
    Augmentation ảnh cho Malignant oversampling (khớp notebook).

    Args:
        img_array      : float32 array [H, W, 3] trong [0, 1]
        rotation_range : độ xoay ngẫu nhiên (weak mode)
        brightness_range: khoảng độ sáng (weak mode)
        zoom_range     : tỷ lệ zoom (weak mode, chưa dùng trong strong)
        strong         : True → augmentation mạnh (dùng khi tạo thêm mẫu Malignant)
                           - Rotation ±30°
                           - Flip ngang + dọc ngẫu nhiên
                           - Saturation & Hue jitter
                           - Contrast jitter
                           - Random crop + resize

    Returns:
        float32 array [H, W, 3] trong [0, 1]
    """
    if img_array is None:
        return None

    img_uint8 = (img_array * 255).astype(np.uint8)
    h, w = img_uint8.shape[:2]

    # ── Rotation ─────────────────────────────────────────────────────────────
    angle = np.random.uniform(-30, 30) if strong else np.random.uniform(-rotation_range, rotation_range)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_uint8 = np.asarray(cv2.warpAffine(img_uint8, M, (w, h), borderMode=cv2.BORDER_REFLECT), dtype=np.uint8)

    # ── Flip (strong only) ────────────────────────────────────────────────────
    if strong:
        if np.random.rand() > 0.5:
            img_uint8 = np.asarray(cv2.flip(img_uint8, 1), dtype=np.uint8)   # horizontal
        if np.random.rand() > 0.5:
            img_uint8 = np.asarray(cv2.flip(img_uint8, 0), dtype=np.uint8)   # vertical

    # ── Brightness ────────────────────────────────────────────────────────────
    factor = np.random.uniform(0.6, 1.4) if strong else np.random.uniform(*brightness_range)
    img_pil = ImageEnhance.Brightness(Image.fromarray(img_uint8)).enhance(factor)

    # ── Saturation + Contrast + Random crop (strong only) ────────────────────
    if strong:
        img_pil = ImageEnhance.Color(img_pil).enhance(np.random.uniform(0.7, 1.3))
        img_pil = ImageEnhance.Contrast(img_pil).enhance(np.random.uniform(0.8, 1.2))

        img_tmp  = np.array(img_pil)
        crop_pct = np.random.uniform(0.85, 1.0)
        crop_h, crop_w = int(h * crop_pct), int(w * crop_pct)
        top  = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        img_tmp = img_tmp[top:top + crop_h, left:left + crop_w]
        img_tmp = cv2.resize(img_tmp, (w, h))
        return img_tmp.astype(np.float32) / 255.0

    return np.array(img_pil).astype(np.float32) / 255.0


# ── Oversampling ──────────────────────────────────────────────────────────────

def oversample_malignant(X_img: np.ndarray,
                         X_tab: np.ndarray,
                         y: np.ndarray,
                         target_ratio: float = 0.25,
                         strong_aug: bool = True,
                         random_seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Oversampling class Malignant bằng augmentation mạnh (khớp notebook).

    Thay thế SMOTE (không phù hợp với ảnh) bằng cách tạo thêm ảnh augmented
    từ các mẫu Malignant hiện có.
    Args:
        X_img        : float32 [N, H, W, 3]
        X_tab        : float32 [N, n_features]
        y            : int32   [N]  (0=Benign, 1=Malignant)
        target_ratio : tỷ lệ Malignant mong muốn sau oversampling
        strong_aug   : dùng augmentation mạnh cho mẫu tạo thêm
        random_seed  : seed cho shuffle cuối

    Returns:
        (X_img_os, X_tab_os, y_os) — đã shuffle ngẫu nhiên
    """
    np.random.seed(random_seed)

    mal_idx = np.where(y == 1)[0]
    ben_idx = np.where(y == 0)[0]
    n_mal, n_ben = len(mal_idx), len(ben_idx)
    current_ratio = n_mal / (n_mal + n_ben)

    print(f"  Trước oversampling: {n_ben:,} Benign | {n_mal:,} Malignant ({current_ratio*100:.1f}%)")

    n_target_mal = int(target_ratio * n_ben / (1 - target_ratio))
    n_to_add = max(0, n_target_mal - n_mal)

    if n_to_add == 0:
        print("  Không cần oversample.")
        return X_img, X_tab, y

    print(f"  Sẽ tạo thêm {n_to_add:,} mẫu Malignant (augmentation {'mạnh' if strong_aug else 'nhẹ'})...")

    new_imgs, new_tabs = [], []
    for i in tqdm(range(n_to_add), desc="  Augmenting Malignant"):
        src_idx = mal_idx[i % n_mal]
        aug_img = augment_image(X_img[src_idx], strong=strong_aug)
        new_imgs.append(aug_img)
        new_tabs.append(X_tab[src_idx])

    new_imgs_arr = np.array(new_imgs, dtype=np.float32)
    new_tabs_arr = np.array(new_tabs, dtype=np.float32)
    new_y    = np.ones(n_to_add, dtype=np.int32)

    X_img_os = np.concatenate([X_img, new_imgs_arr], axis=0)
    X_tab_os = np.concatenate([X_tab, new_tabs_arr], axis=0)
    y_os     = np.concatenate([y, new_y], axis=0)

    # Shuffle
    perm = np.random.permutation(len(y_os))
    X_img_os, X_tab_os, y_os = X_img_os[perm], X_tab_os[perm], y_os[perm]

    n_mal_new = int(np.sum(y_os == 1))
    n_ben_new = int(np.sum(y_os == 0))
    print(f"  Sau oversampling : {n_ben_new:,} Benign | {n_mal_new:,} Malignant "
          f"({n_mal_new/(n_mal_new+n_ben_new)*100:.1f}%)")

    return X_img_os, X_tab_os, y_os


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BƯỚC 3: Tạo features array + splits + oversampling → Local/DVC")
    print(f"  Bucket          : {DATA_DIR}")
    print(f"  Oversample ratio: {OVERSAMPLE_RATIO}")
    print("=" * 60)

    split_info_path = os.path.join(DATA_DIR, "splits/split_info.json")
    if os.path.exists(split_info_path):
        print(f"\n Tìm thấy '{split_info_path}'.")
        print("Dataloader đã chạy thành công trước đó. BỎ QUA (SKIPPED).")
        return

    df       = pd.read_csv(os.path.join(DATA_DIR, "preprocessed/metadata_clean.csv"))
    with open(os.path.join(DATA_DIR, "preprocessed/encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    feature_cols = encoders["feature_cols"]

    # ── Kiểm tra ảnh có trên Local/DVC ───────────────────────────────────────────────
    print("\nKiểm tra ảnh đã preprocessed trên local...")
    img_dir = os.path.join(DATA_DIR, "preprocessed/images")
    existing_keys = os.listdir(img_dir) if os.path.exists(img_dir) else []
    existing_ids  = {k.replace(".png", "") for k in existing_keys}
    df_available  = df[df["isic_id"].isin(existing_ids)].reset_index(drop=True)
    print(f"  Ảnh có sẵn: {len(existing_ids):,}")
    print(f"  Mẫu khớp  : {len(df_available):,}")

    # ── Build raw arrays ──────────────────────────────────────────────────────
    n_samples = len(df_available)
    print(f"\nĐang tải ảnh + build arrays (pre-allocated cho {n_samples:,} mẫu)...")

    X_tabular = np.zeros((n_samples, len(feature_cols)), dtype=np.float32)
    X_images  = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    y_labels  = np.zeros(n_samples, dtype=np.int32)

    valid_count = 0
    for _, row in tqdm(df_available.iterrows(), total=n_samples, desc="Loading images"):
        img = load_image(row["isic_id"])
        if img is None:
            continue
        X_tabular[valid_count] = row[feature_cols].values.astype(np.float32)
        X_images[valid_count]  = img
        y_labels[valid_count]  = int(row["target"])
        valid_count += 1

    X_tabular = X_tabular[:valid_count]
    X_images  = X_images[:valid_count]
    y_labels  = y_labels[:valid_count]

    print(f"\nArrays built:")
    print(f"  X_tabular : {X_tabular.shape}")
    print(f"  X_images  : {X_images.shape}")
    print(f"  y_labels  : {y_labels.shape}  "
          f"(Mal={int(y_labels.sum())}, Ben={int((y_labels == 0).sum())})")
    print(f"  Tỷ lệ Malignant: {np.mean(y_labels)*100:.2f}%")

    # Lưu features gốc
    np.save(os.path.join(DATA_DIR, "features/X_tabular.npy"), X_tabular)
    np.save(os.path.join(DATA_DIR, "features/X_images.npy"), X_images)
    np.save(os.path.join(DATA_DIR, "features/y_labels.npy"), y_labels)

    # ── Stratified split 64 / 16 / 20 ────────────────────────────────────────
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

    split_info = {
        "random_seed":      RANDOM_SEED,
        "oversample_ratio": OVERSAMPLE_RATIO,
        "splits": {},
    }

    print("\n" + "=" * 60)
    print("STRATIFIED SPLITS")
    print("=" * 60)

    X_tab_train = X_tab_val = X_tab_test = None
    X_img_train = X_img_val = X_img_test = None
    y_train = y_val = y_test = None

    for name, (idx_s, prefix) in splits.items():
        xs = X_tabular[idx_s]
        xi = X_images[idx_s]
        ys = y_labels[idx_s]

        np.save(os.path.join(DATA_DIR, f"{prefix}X_tab_{name}.npy"), xs)
        np.save(os.path.join(DATA_DIR, f"{prefix}X_img_{name}.npy"), xi)
        np.save(os.path.join(DATA_DIR, f"{prefix}y_{name}.npy"), ys)

        split_info["splits"][name] = {
            "total":     int(len(ys)),
            "malignant": int(ys.sum()),
            "benign":    int((ys == 0).sum()),
            "ratio_mal": round(float(ys.mean()), 4),
        }
        print(f"  {name:5s}: {len(ys):>6,} mẫu | "
              f"Mal={int(ys.sum())} ({100*ys.mean():.1f}%)")

        if name == "train":
            X_tab_train, y_train = xs, ys
        elif name == "val":
            X_tab_val, y_val = xs, ys
        elif name == "test":
            X_tab_test, y_test = xs, ys

        # Dọn dẹp RAM ngay sau khi lưu
        del xi

    import gc
    del X_images
    gc.collect()

    # ── Oversampling trên tập TRAIN ───────────────────────────────────────────
    # Chỉ oversample train — val và test giữ nguyên phân phối thật
    # để đánh giá mô hình trên dữ liệu thực tế
    print("\n" + "=" * 60)
    print("OVERSAMPLING MALIGNANT — CHỈ TRÊN TẬP TRAIN")
    print("=" * 60)
    print(f"  target_ratio = {OVERSAMPLE_RATIO} (giảm từ 0.35 → 0.25 so với bản cũ, tránh overfit)")
    print(f"  strong augmentation = True")
    print(f"  Lý do: augmentation đa dạng hơn SMOTE, phù hợp hơn với dữ liệu ảnh")

    # Load lại X_img_train dưới dạng memmap để tiết kiệm RAM
    X_img_train = np.load(os.path.join(DATA_DIR, "splits/train/X_img_train.npy"), mmap_mode="r")

    X_img_train_os, X_tab_train_os, y_train_os = oversample_malignant(
        X_img_train, X_tab_train, y_train,
        target_ratio=OVERSAMPLE_RATIO,
        strong_aug=True,
        random_seed=RANDOM_SEED,
    )

    # Lưu tập train đã oversampled
    np.save(os.path.join(DATA_DIR, "splits/train/X_tab_train_os.npy"), X_tab_train_os)
    np.save(os.path.join(DATA_DIR, "splits/train/X_img_train_os.npy"), X_img_train_os)
    np.save(os.path.join(DATA_DIR, "splits/train/y_train_os.npy"), y_train_os)

    # Ghi thông tin oversampling vào split_info
    split_info["splits"]["train_os"] = {
        "total":     int(len(y_train_os)),
        "malignant": int(y_train_os.sum()),
        "benign":    int((y_train_os == 0).sum()),
        "ratio_mal": round(float(y_train_os.mean()), 4),
    }

    # ── Class weight để dùng khi huấn luyện ──────────────────────────────────
    #  Cải tiến: ×1.2 thay vì ×1.5 để tránh triple-penalty bias
    # (Focal Loss + Oversampling + ClassWeight×1.5 → bias nặng về Malignant)
    n_neg = int(np.sum(y_train_os == 0))
    n_pos = int(np.sum(y_train_os == 1))
    class_weight_dict = {
        0: 1.0,
        1: round((n_neg / n_pos) * 1.2, 4),   # ×1.2 thay vì ×1.5
    }
    split_info["class_weight"] = class_weight_dict

    print(f"\n  Class weights (×1.2): {class_weight_dict}")
    print(f"  Lý do ×1.2: tránh triple-penalty bias (Focal + Oversample + ClassWeight)")

    # ── Ghi split_info.json ────────────────────────────────────────────
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HOÀN THÀNH — TÓM TẮT")
    print("=" * 60)
    print(f"  train     : {len(y_train):>6,} mẫu  (gốc, trước oversampling)")
    print(f"  train_os  : {len(y_train_os):>6,} mẫu  (sau oversampling {OVERSAMPLE_RATIO:.0%} Malignant)")
    print(f"  val       : {len(y_val):>6,} mẫu  (không oversample)")
    print(f"  test      : {len(y_test):>6,} mẫu  (không oversample)")
    print("\n  Split info → {split_info_path}")

    print("\nĐồng bộ thư mục lên S3 Output Bucket (thay thế DVC)...")

    def upload_worker(local_path):
        rel_path = os.path.relpath(local_path, DATA_DIR)
        # convert backslashes to forward slashes for S3
        s3_key = rel_path.replace("\\", "/")
        s3_utils.upload_file(local_path, s3_key)
        
    upload_list = []
    for d in ["features", "splits"]:
        for root, dirs, files in os.walk(os.path.join(DATA_DIR, d)):
            for f in files:
                upload_list.append(os.path.join(root, f))
                
    print(f"Bắt đầu upload {len(upload_list)} file array/json bằng đa luồng...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(upload_worker, upload_list), total=len(upload_list), desc="Uploading"))

    print("\nBước 3 hoàn thành!")


if __name__ == "__main__":
    main()
