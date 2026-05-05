"""
download_dataset.py — Bước 1: Tải và chuẩn bị dataset ISIC 2024 từ AWS S3

Pipeline theo notebook multimodal_skin_isic2024_efficientnetB3:
  [1] Tải train-metadata.csv + train-image.hdf5 từ S3
  [2] Kiểm tra tính toàn vẹn dữ liệu (CSV ↔ HDF5)
  [3] Tiền xử lý metadata:
        - Chuẩn hóa tên cột
        - Điền giá trị thiếu (median cho số, mode cho chuỗi)
        - Clip outliers bằng IQR (chỉ các cột không phải 'target')
  [4] Trích xuất ảnh có chọn lọc từ HDF5 (cân bằng class):
        - Tất cả Malignant (target=1)
        - Sample ngẫu nhiên Benign   (target=0)
  [5] Lưu metadata sạch + thống kê tóm tắt

Đầu ra:
  DATA_DIR/train-metadata.csv          ← Metadata gốc
  DATA_DIR/train-image.hdf5            ← HDF5 ảnh gốc
  DATA_DIR/metadata_clean.csv          ← Metadata sau tiền xử lý
  DATA_DIR/images/                     ← Ảnh đã trích xuất (*.jpg)
"""

import os
import zipfile

import boto3
import h5py
import numpy as np
import pandas as pd

# ─── Cấu hình ──────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/data/raw")

# AWS credentials — ưu tiên biến môi trường
AWS_ACCESS_KEY_ID     = os.environ.get("AKIATTX4CRCOWN6O67VG", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("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", "")
AWS_SESSION_TOKEN     = os.environ.get("AWS_SESSION_TOKEN", None)
AWS_REGION            = os.environ.get("AWS_REGION", "ap-southeast-1")

# S3 paths
S3_INPUT_BUCKET  = os.environ.get("S3_INPUT_BUCKET",  "kltn-isic-2024-challenge")
S3_INPUT_PREFIX  = os.environ.get("S3_INPUT_PREFIX",  "isic-2024-challenge")

# Tham số sampling để cân bằng class (theo notebook)
N_BENIGN_SAMPLE  = int(os.environ.get("N_BENIGN_SAMPLE", "4000"))   # số Benign lấy mẫu
RANDOM_SEED      = int(os.environ.get("RANDOM_SEED",      "42"))

# Đường dẫn local
LOCAL_METADATA   = os.path.join(DATA_DIR, "train-metadata.csv")
LOCAL_HDF5       = os.path.join(DATA_DIR, "train-image.hdf5")
LOCAL_CLEAN_CSV  = os.path.join(DATA_DIR, "metadata_clean.csv")
IMAGE_DIR        = os.path.join(DATA_DIR, "images")

# Xác nhận credentials
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError(
        "Chưa đặt biến môi trường AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY!\n"
        "  export AWS_ACCESS_KEY_ID=<key>\n"
        "  export AWS_SECRET_ACCESS_KEY=<secret>"
    )

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ─── Khởi tạo S3 client ────────────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    region_name='ap-southeast-1',
    aws_access_key_id='AKIATTX4CRCOWN6O67VG',
    aws_secret_access_key='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',
    aws_session_token=None,
)


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 1 — Tải dữ liệu từ S3
# ═══════════════════════════════════════════════════════════════════════════
def _download_if_missing(s3_key: str, local_path: str, label: str) -> None:
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1_048_576
        print(f"  [skip] {label} đã tồn tại ({size_mb:.1f} MB): {local_path}")
        return
    print(f"  Đang tải {label} từ s3://{S3_INPUT_BUCKET}/{s3_key} ...")
    s3.download_file(S3_INPUT_BUCKET, s3_key, local_path)
    size_mb = os.path.getsize(local_path) / 1_048_576
    print(f"   Đã lưu {label} ({size_mb:.1f} MB) → {local_path}")


print("\n[1/4] Tải dataset từ S3 ...")
_download_if_missing(f"{S3_INPUT_PREFIX}/train-metadata.csv", LOCAL_METADATA, "train-metadata.csv")
_download_if_missing(f"{S3_INPUT_PREFIX}/train-image.hdf5",  LOCAL_HDF5,     "train-image.hdf5 (file lớn)")


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — Kiểm tra tính toàn vẹn & đối chiếu CSV ↔ HDF5
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/4] Kiểm tra dữ liệu ...")

df_raw = pd.read_csv(LOCAL_METADATA)
print(f"  CSV  : {len(df_raw):,} dòng, {len(df_raw.columns)} cột")

with h5py.File(LOCAL_HDF5, "r") as hf:
    hdf5_ids = set(hf.keys())
print(f"  HDF5 : {len(hdf5_ids):,} ảnh")

csv_ids    = set(df_raw["isic_id"].astype(str))
only_csv   = csv_ids - hdf5_ids
only_hdf5  = hdf5_ids - csv_ids
common_ids = csv_ids & hdf5_ids

print(f"  isic_id chỉ có trong CSV  : {len(only_csv):,}")
print(f"  isic_id chỉ có trong HDF5 : {len(only_hdf5):,}")
print(f"  isic_id khớp cả hai        : {len(common_ids):,}")

# Chỉ giữ các mẫu có ảnh trong HDF5
df_raw = df_raw[df_raw["isic_id"].astype(str).isin(common_ids)].reset_index(drop=True)
print(f"  → Sau lọc: {len(df_raw):,} mẫu hợp lệ")

# Phân phối nhãn ban đầu
n_mal = (df_raw["target"] == 1).sum()
n_ben = (df_raw["target"] == 0).sum()
print(f"  Benign (0)    : {n_ben:,}")
print(f"  Malignant (1) : {n_mal:,}  ({n_mal/len(df_raw)*100:.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 3 — Tiền xử lý metadata  (theo hàm preprocess_csv_data của notebook)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/4] Tiền xử lý metadata ...")


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df_p = df.copy()

    # 3a. Chuẩn hóa tên cột
    df_p.columns = (
        df_p.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+",          "_", regex=True)
        .str.replace(r"[^a-z0-9_]",  "_", regex=True)
        .str.replace(r"_+",          "_", regex=True)
        .str.strip("_")
    )

    # 3b. Điền giá trị thiếu
    missing_total = df_p.isnull().sum().sum()
    if missing_total:
        print(f"  Dữ liệu thiếu: {missing_total:,} ô")
        for col in df_p.columns:
            n_miss = df_p[col].isnull().sum()
            if n_miss == 0:
                continue
            pct = n_miss / len(df_p) * 100
            if df_p[col].dtype in ("float64", "int64"):
                fill_val = df_p[col].median()
                strategy = f"median={fill_val:.3g}"
            else:
                mode_vals = df_p[col].mode()
                fill_val  = mode_vals[0] if len(mode_vals) else "unknown"
                strategy  = f"mode='{fill_val}'"
            df_p[col].fillna(fill_val, inplace=True)
            print(f"    {col}: {n_miss:,} thiếu ({pct:.1f}%) → điền {strategy}")
    else:
        print("  Không có dữ liệu thiếu.")

    # 3c. Clip outliers bằng IQR (bỏ qua cột target và cột id)
    skip_cols   = {"target", "isic_id"}
    numeric_cols = [
        c for c in df_p.select_dtypes(include=["float64", "int64"]).columns
        if c not in skip_cols
    ]
    n_clipped = 0
    for col in numeric_cols:
        q1, q3  = df_p[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df_p[col] < lo) | (df_p[col] > hi)).sum()
        if outliers:
            df_p[col] = df_p[col].clip(lo, hi)
            n_clipped += outliers
    print(f"  Đã clip {n_clipped:,} outlier trong {len(numeric_cols)} cột số (IQR ×1.5)")

    return df_p


df_clean = preprocess_metadata(df_raw)
df_clean.to_csv(LOCAL_CLEAN_CSV, index=False)
print(f"   Đã lưu metadata sạch → {LOCAL_CLEAN_CSV}")


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 4 — Trích xuất ảnh có chọn lọc từ HDF5 (cân bằng class)
#           Theo logic notebook: tất cả Malignant + sample Benign ngẫu nhiên
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/4] Trích xuất ảnh từ HDF5 (cân bằng class) ...")

mal_ids = df_clean[df_clean["target"] == 1]["isic_id"].astype(str).tolist()
ben_ids = df_clean[df_clean["target"] == 0]["isic_id"].astype(str).tolist()

n_mal_extract = len(mal_ids)
n_ben_extract = min(N_BENIGN_SAMPLE, len(ben_ids))

rng = np.random.default_rng(RANDOM_SEED)
selected_ben = rng.choice(ben_ids, size=n_ben_extract, replace=False).tolist()
selected_ids = set(mal_ids + selected_ben)

print(f"  Malignant : {n_mal_extract:,} (100% — không bỏ sót)")
print(f"  Benign    : {n_ben_extract:,} (sample ngẫu nhiên, seed={RANDOM_SEED})")
print(f"  Tổng      : {len(selected_ids):,}  "
      f"| Tỷ lệ Malignant ≈ {n_mal_extract/len(selected_ids)*100:.1f}%")

# Kiểm tra ảnh đã trích xuất trước đó
existing = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")}
to_extract = selected_ids - existing
print(f"  Đã có sẵn : {len(existing):,} ảnh  |  Cần trích xuất thêm: {len(to_extract):,}")

extracted = errors = 0
if to_extract:
    with h5py.File(LOCAL_HDF5, "r") as hf:
        for isic_id in hf.keys():
            if isic_id not in to_extract:
                continue
            try:
                img_bytes   = hf[isic_id][()]
                output_path = os.path.join(IMAGE_DIR, f"{isic_id}.jpg")
                with open(output_path, "wb") as fp:
                    fp.write(img_bytes)
                extracted += 1
                if extracted % 500 == 0:
                    print(f"    Đã trích xuất: {extracted:,}/{len(to_extract):,}")
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"    Lỗi {isic_id}: {exc}")
    print(f"   Hoàn thành: {extracted:,} ảnh mới | Lỗi: {errors}")
else:
    print("  [skip] Tất cả ảnh đã tồn tại.")

# ─── Xác minh phân phối sau trích xuất ────────────────────────────────────
available_ids = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")}
df_check      = df_clean[df_clean["isic_id"].astype(str).isin(available_ids)]
n_ben_ok  = (df_check["target"] == 0).sum()
n_mal_ok  = (df_check["target"] == 1).sum()

print("\n─── Tóm tắt ─────────────────────────────────────────────────")
print(f"  Metadata sạch  : {len(df_clean):,} dòng → {LOCAL_CLEAN_CSV}")
print(f"  Ảnh đã trích   : {len(available_ids):,} files → {IMAGE_DIR}/")
print(f"    Benign (0)   : {n_ben_ok:,}")
print(f"    Malignant (1): {n_mal_ok:,}  ({n_mal_ok/(n_ben_ok+n_mal_ok)*100:.1f}%)")
print("─────────────────────────────────────────────────────────────")
print(" Chuẩn bị dataset hoàn thành!")
