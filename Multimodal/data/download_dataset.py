"""
prepare_data.py — Bước 1: Kiểm tra dữ liệu thô trên S3 input bucket

Input  : s3://kltn-isic-2024-challenge/isic-2024-challenge/train-metadata.csv
         s3://kltn-isic-2024-challenge/isic-2024-challenge/train-image.hdf5
Output : s3://kltn-isic-2024-colab/raw/metadata.csv   (copy metadata gốc)
         In ra manifest JSON xác nhận dataset sẵn sàng
"""
import io
import json
import os
import tempfile
import h5py
import pandas as pd
from s3_utils import (
    get_s3_client,
    download_bytes,
    upload_bytes,
    save_csv,
    S3_INPUT_BUCKET, S3_INPUT_PREFIX, S3_OUTPUT_BUCKET,
)

def main():
    print("=" * 60)
    print("BƯỚC 1: Kiểm tra dataset ISIC 2024 trên S3")
    print(f"  Input : s3://{S3_INPUT_BUCKET}/{S3_INPUT_PREFIX}/")
    print(f"  Output: s3://{S3_OUTPUT_BUCKET}/raw/")
    print("=" * 60)

    # ── 1. Kiểm tra & copy metadata CSV ─────────────────────────────────
    csv_key = f"{S3_INPUT_PREFIX}/train-metadata.csv"
    print(f"\n[1/3] Đọc metadata CSV...")
    csv_bytes = download_bytes(csv_key, bucket=S3_INPUT_BUCKET)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    print(f"  Tổng mẫu  : {len(df):,}")
    print(f"  Malignant : {(df['target']==1).sum():,}")
    print(f"  Benign    : {(df['target']==0).sum():,}")
    print(f"  Cột       : {list(df.columns)[:8]}...")

    # Copy sang output bucket để các bước sau đọc từ một nơi thống nhất
    upload_bytes(csv_bytes, "raw/metadata.csv", bucket=S3_OUTPUT_BUCKET)

    # ── 2. Kiểm tra HDF5 (chỉ đọc header, không tải toàn bộ) ────────────
    hdf5_key = f"{S3_INPUT_PREFIX}/train-image.hdf5"
    print(f"\n[2/3] Kiểm tra HDF5 image file...")
    s3 = get_s3_client()
    resp = s3.head_object(Bucket=S3_INPUT_BUCKET, Key=hdf5_key)
    hdf5_size_gb = resp["ContentLength"] / 1e9
    print(f"  s3://{S3_INPUT_BUCKET}/{hdf5_key}")
    print(f"  Kích thước: {hdf5_size_gb:.2f} GB")

    # ── 3. Ghi manifest ─────────────────────────────────────────────────
    manifest = {
        "s3_input_bucket":  S3_INPUT_BUCKET,
        "s3_input_prefix":  S3_INPUT_PREFIX,
        "s3_output_bucket": S3_OUTPUT_BUCKET,
        "csv_rows":    len(df),
        "malignant":   int((df["target"] == 1).sum()),
        "benign":      int((df["target"] == 0).sum()),
        "hdf5_size_gb": round(hdf5_size_gb, 3),
        "status": "ready",
    }
    upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        "raw/manifest.json",
        bucket=S3_OUTPUT_BUCKET,
    )
    print(f"\n[3/3] Manifest lưu → s3://{S3_OUTPUT_BUCKET}/raw/manifest.json")
    print("\nBước 1 hoàn thành!")

if __name__ == "__main__":
    main()
