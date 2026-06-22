"""
prepare_data.py — Bước 1: Fetch dữ liệu thô từ S3 (ISIC Challenge) xuống PVC cục bộ.

Kịch bản này dùng boto3 tải trực tiếp file từ bucket ngoài (kltn-isic-2024-challenge)
vào thư mục DATA_DIR (/data/raw trên PVC) để mồi dữ liệu cho các bước sau.
"""
import os
import json
import boto3
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data/raw")
S3_INPUT_BUCKET = os.environ.get("S3_INPUT_BUCKET", "kltn-isic-2024-challenge")
S3_INPUT_PREFIX = os.environ.get("S3_INPUT_PREFIX", "isic-2024-challenge")

os.makedirs(DATA_DIR, exist_ok=True)

def get_s3_client():
    # Sử dụng credentials từ môi trường K8s/Argo
    return boto3.client('s3')

def main():
    print("=" * 60)
    print("BƯỚC 1: Fetch dataset ISIC 2024 từ S3 -> PVC")
    print(f"  Bucket: {S3_INPUT_BUCKET}")
    print(f"  Prefix: {S3_INPUT_PREFIX}")
    print(f"  Lưu tại: {DATA_DIR}")
    print("=" * 60)

    s3 = get_s3_client()

    # ── 1. Tải & copy metadata CSV ─────────────────────────────────
    csv_key = f"{S3_INPUT_PREFIX}/train-metadata.csv"
    csv_in_path = os.path.join(DATA_DIR, "train-metadata.csv")
    
    print(f"\n[1/3] Tải metadata CSV từ s3://{S3_INPUT_BUCKET}/{csv_key}...")
    if not os.path.exists(csv_in_path):
        s3.download_file(S3_INPUT_BUCKET, csv_key, csv_in_path)
        print(f"  Đã tải xong: {csv_in_path}")
    else:
        print(f"  File đã tồn tại, bỏ qua tải: {csv_in_path}")

    df = pd.read_csv(csv_in_path)
    print(f"  Tổng mẫu  : {len(df):,}")
    print(f"  Malignant : {(df['target']==1).sum():,}")
    print(f"  Benign    : {(df['target']==0).sum():,}")

    # Lưu thêm 1 bản metadata.csv theo format chuẩn của pipeline
    csv_out_path = os.path.join(DATA_DIR, "metadata.csv")
    df.to_csv(csv_out_path, index=False)
    print(f"  Đã tạo bản sao metadata.csv tại {csv_out_path}")

    # ── 2. Tải HDF5 ────────────────────────────────────────────────
    hdf5_key = f"{S3_INPUT_PREFIX}/train-image.hdf5"
    hdf5_path = os.path.join(DATA_DIR, "train-image.hdf5")
    
    print(f"\n[2/3] Tải HDF5 image file từ s3://{S3_INPUT_BUCKET}/{hdf5_key}...")
    if not os.path.exists(hdf5_path):
        s3.download_file(S3_INPUT_BUCKET, hdf5_key, hdf5_path)
        print(f"  Đã tải xong HDF5: {hdf5_path}")
    else:
        print(f"  File HDF5 đã tồn tại, bỏ qua tải: {hdf5_path}")

    hdf5_size_gb = os.path.getsize(hdf5_path) / 1e9
    print(f"  Kích thước thực tế: {hdf5_size_gb:.2f} GB")

    # ── 3. Ghi manifest ─────────────────────────────────────────────────
    manifest = {
        "csv_rows": len(df),
        "malignant": int((df["target"] == 1).sum()),
        "benign": int((df["target"] == 0).sum()),
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
