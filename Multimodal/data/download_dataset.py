"""
prepare_data.py — Bước 1: Kiểm tra dữ liệu thô trên S3 input bucket

Input  : s3://kltn-isic-2024-challenge/isic-2024-challenge/train-metadata.csv
         s3://kltn-isic-2024-challenge/isic-2024-challenge/train-image.hdf5
Output : s3://kltn-isic-2024-colab/raw/metadata.csv   (copy metadata gốc)
         In ra manifest JSON xác nhận dataset sẵn sàng
"""
import os
import json
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data/raw")
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("BƯỚC 1: Kiểm tra dataset ISIC 2024 (Local via DVC)")
    print(f"  Thư mục: {DATA_DIR}")
    print("=" * 60)

    # ── 1. Kiểm tra & copy metadata CSV ─────────────────────────────────
    csv_in_path = os.path.join(DATA_DIR, "train-metadata.csv")
    print(f"\n[1/3] Đọc metadata CSV từ {csv_in_path}...")

    if not os.path.exists(csv_in_path):
        raise FileNotFoundError(f"Không tìm thấy {csv_in_path}. Hãy đảm bảo DVC đã pull dữ liệu.")

    df = pd.read_csv(csv_in_path)
    print(f"  Tổng mẫu  : {len(df):,}")
    print(f"  Malignant : {(df['target']==1).sum():,}")
    print(f"  Benign    : {(df['target']==0).sum():,}")
    print(f"  Cột       : {list(df.columns)[:8]}...")

    # Copy sang output để các bước sau đọc từ một nơi thống nhất
    csv_out_path = os.path.join(DATA_DIR, "metadata.csv")
    df.to_csv(csv_out_path, index=False)
    print(f"  Đã lưu metadata.csv tại {csv_out_path}")

    # ── 2. Kiểm tra HDF5 ────────────────────────────────────────────────
    hdf5_path = os.path.join(DATA_DIR, "train-image.hdf5")
    print(f"\n[2/3] Kiểm tra HDF5 image file tại {hdf5_path}...")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Không tìm thấy {hdf5_path}. Hãy đảm bảo DVC đã pull dữ liệu.")

    hdf5_size_gb = os.path.getsize(hdf5_path) / 1e9
    print(f"  Kích thước: {hdf5_size_gb:.2f} GB")

    # ── 3. Ghi manifest ─────────────────────────────────────────────────
    manifest = {
        "csv_rows":    len(df),
        "malignant":   int((df["target"] == 1).sum()),
        "benign":      int((df["target"] == 0).sum()),
        "hdf5_size_gb": round(hdf5_size_gb, 3),
        "status": "ready",
    }
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[3/3] Manifest lưu → {manifest_path}")
    print("\nBước 1 hoàn thành!")

if __name__ == "__main__":
    main()
