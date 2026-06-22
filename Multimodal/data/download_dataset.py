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

    # Danh sách các file cần tải từ S3
    files_to_download = [
        "train-metadata.csv",
        "train-image.hdf5",
        "test-metadata.csv",
        "test-image.hdf5",
        "sample_submission.csv"
    ]

    for file_name in files_to_download:
        s3_key = f"{S3_INPUT_PREFIX}/{file_name}"
        local_path = os.path.join(DATA_DIR, file_name)

        print(f"\n[*] Đang kiểm tra s3://{S3_INPUT_BUCKET}/{s3_key}...")
        if not os.path.exists(local_path):
            try:
                print(f"    -> Đang tải về {local_path}...")
                s3.download_file(S3_INPUT_BUCKET, s3_key, local_path)
                print("    -> Tải thành công!")
            except Exception as e:
                print(f"    -> Bỏ qua (không tìm thấy trên S3 hoặc lỗi): {e}")
        else:
            print(f"    -> File đã tồn tại ở PVC, bỏ qua tải: {local_path}")

    # ── 1. Đọc và phân tích metadata CSV ─────────────────────────────────
    csv_in_path = os.path.join(DATA_DIR, "train-metadata.csv")
    if os.path.exists(csv_in_path):
        df = pd.read_csv(csv_in_path)
        print(f"\nThống kê train-metadata:")
        print(f"  Tổng mẫu  : {len(df):,}")
        print(f"  Malignant : {(df['target']==1).sum():,}")
        print(f"  Benign    : {(df['target']==0).sum():,}")

        # Lưu thêm 1 bản metadata.csv theo format chuẩn của pipeline
        csv_out_path = os.path.join(DATA_DIR, "metadata.csv")
        df.to_csv(csv_out_path, index=False)
        print(f"  Đã tạo bản sao metadata.csv tại {csv_out_path}")
    else:
        df = []  # fallback

    # ── 2. Kiểm tra HDF5 ────────────────────────────────────────────────
    hdf5_path = os.path.join(DATA_DIR, "train-image.hdf5")
    if os.path.exists(hdf5_path):
        hdf5_size_gb = os.path.getsize(hdf5_path) / 1e9
        print(f"\nThống kê train-image.hdf5:")
        print(f"  Kích thước thực tế: {hdf5_size_gb:.2f} GB")
    else:
        hdf5_size_gb = 0

    # ── 3. Ghi manifest ─────────────────────────────────────────────────
    manifest = {
        "csv_rows": len(df) if isinstance(df, pd.DataFrame) else 0,
        "malignant": int((df["target"] == 1).sum()) if isinstance(df, pd.DataFrame) else 0,
        "benign": int((df["target"] == 0).sum()) if isinstance(df, pd.DataFrame) else 0,
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
