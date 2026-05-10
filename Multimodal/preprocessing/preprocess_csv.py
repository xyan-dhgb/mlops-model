"""
preprocess_csv.py — Bước 2b: Tiền xử lý metadata CSV

Đọc : s3://kltn-isic-2024-colab/raw/metadata.csv
Ghi :
  s3://kltn-isic-2024-colab/preprocessed/metadata_clean.csv
  s3://kltn-isic-2024-colab/preprocessed/encoders.pkl

Pipeline (khớp notebook cell 19 + cell 32):
  - Chuẩn hóa tên cột
  - Impute median (numeric) / mode (categorical)
  - Clip outlier IQR×1.5
  - LabelEncode: sex, anatom_site_general
  - StandardScaler cho tất cả feature
  - Loại exclude_cols (khớp cell 32 của notebook)
"""
import os
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from s3_utils import (
    load_csv, save_csv, save_pkl, S3_OUTPUT_BUCKET,
)

# Cột loại trừ — khớp notebook cell 32
EXCLUDE_COLS = [
    "target", "isic_id", "patient_id", "attribution", "copyright_license",
    "image_type", "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4",
    "iddx_5", "mel_mitotic_index", "mel_thick_mm", "lesion_id",
]

CAT_COLS = ["sex", "anatom_site_general"]   # encode riêng


def main():
    print("=" * 60)
    print("BƯỚC 2b: Tiền xử lý CSV")
    print(f"  Bucket: s3://{S3_OUTPUT_BUCKET}/preprocessed/")
    print("=" * 60)

    df = load_csv("raw/metadata.csv", bucket=S3_OUTPUT_BUCKET)
    print(f"Shape gốc: {df.shape}")

    # 1. Chuẩn hóa tên cột (cell 19)
    df.columns = (df.columns
                  .str.strip().str.lower()
                  .str.replace(r'\s+', '_', regex=True)
                  .str.replace(r'[^a-z0-9_]', '_', regex=True)
                  .str.replace(r'_+', '_', regex=True)
                  .str.strip('_'))

    # 2. Xác định cột feature (khớp cell 32)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_num  = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    feature_cat  = [c for c in CAT_COLS if c in df.columns]

    print(f"Numeric features : {len(feature_num)}")
    print(f"Categorical feats: {feature_cat}")

    # 3. Impute (cell 32: SimpleImputer median)
    print("\nImpute missing values...")
    imputer = SimpleImputer(strategy="median")
    df[feature_num] = imputer.fit_transform(df[feature_num])

    for col in feature_cat:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown",
                       inplace=True)

    # 4. Clip outlier IQR (cell 19)
    print("Clip outlier IQR×1.5...")
    for col in feature_num:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # 5. LabelEncode categorical (cell 32)
    label_encoders = {}
    for col in feature_cat:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        if col not in feature_num:
            feature_num.append(col)

    all_feature_cols = feature_num  # numeric + encoded categorical

    # 6. StandardScaler
    print("StandardScaler...")
    scaler = StandardScaler()
    df[all_feature_cols] = scaler.fit_transform(df[all_feature_cols])

    # 7. Lưu kết quả lên S3
    save_csv(df, "preprocessed/metadata_clean.csv", bucket=S3_OUTPUT_BUCKET)

    encoders_obj = {
        "scaler":         scaler,
        "label_encoders": label_encoders,
        "imputer":        imputer,
        "feature_cols":   all_feature_cols,
    }
    save_pkl(encoders_obj, "preprocessed/encoders.pkl", bucket=S3_OUTPUT_BUCKET)

    print(f"\nShape cuối: {df.shape}")
    print(f"Feature cols: {len(all_feature_cols)}")
    print(f"Lưu → s3://{S3_OUTPUT_BUCKET}/preprocessed/metadata_clean.csv")
    print(f"Lưu → s3://{S3_OUTPUT_BUCKET}/preprocessed/encoders.pkl")
    print("\nBước 2b hoàn thành!")


if __name__ == "__main__":
    main()
