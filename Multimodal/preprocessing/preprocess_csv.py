"""
preprocess_csv.py — Bước 2b: Tiền xử lý dữ liệu bảng (tabular)
Pipeline: Load CSV → Impute → Outlier clip → Label Encode → StandardScaler → Lưu pkl
Đầu vào : /data/raw/train-metadata.csv
Đầu ra  : /data/processed/tabular_processed.pkl
           /data/processed/encoders.pkl
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DIR       = os.environ.get("RAW_DIR", "/data/raw")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/data/processed")

CSV_PATH   = os.path.join(RAW_DIR, "train-metadata.csv")
OUTPUT_TAB = os.path.join(PROCESSED_DIR, "tabular_processed.pkl")
OUTPUT_ENC = os.path.join(PROCESSED_DIR, "encoders.pkl")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Cột loại trừ khỏi xử lý feature ────────────────────────────────
EXCLUDE_COLS  = ["isic_id", "target"]
CATEGORICAL_COLS = ["sex", "anatom_site_general"]  # điều chỉnh nếu cần


def main():
    print(f"Đọc CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Shape gốc: {df.shape}")

    # 1. Chuẩn hóa tên cột
    df.columns = (df.columns
                  .str.strip().str.lower()
                  .str.replace(r'\s+', '_', regex=True)
                  .str.replace(r'[^a-z0-9_]', '_', regex=True)
                  .str.strip('_'))

    # 2. Impute giá trị thiếu
    print("\nImpute giá trị thiếu...")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object"]).columns.tolist()

    feature_numeric = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    feature_cat     = [c for c in cat_cols     if c not in EXCLUDE_COLS]

    for col in feature_numeric:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in feature_cat:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if len(mode_val) else "unknown", inplace=True)

    # 3. Clip outlier (IQR) cho cột số — loại bỏ target
    print("Clip outlier bằng IQR...")
    for col in feature_numeric:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # 4. Label Encoding cho cột categorical
    print("Label Encoding...")
    label_encoders = {}
    for col in feature_cat:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 5. StandardScaler cho tất cả cột số feature
    all_feature_cols = feature_numeric + feature_cat
    print(f"StandardScaler cho {len(all_feature_cols)} cột feature...")
    scaler = StandardScaler()
    df[all_feature_cols] = scaler.fit_transform(df[all_feature_cols])

    # 6. Lưu kết quả
    df.to_pickle(OUTPUT_TAB)
    print(f"Lưu tabular_processed.pkl → {OUTPUT_TAB}")

    encoders = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_cols": all_feature_cols,
    }
    with open(OUTPUT_ENC, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Lưu encoders.pkl → {OUTPUT_ENC}")

    print(f"\nHoàn thành tiền xử lý CSV! Shape cuối: {df.shape}")


if __name__ == "__main__":
    main()
