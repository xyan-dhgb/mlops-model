"""
dataloader.py — Bước 3: Stratified split 64%/16%/20%
Đầu vào : /data/processed/tabular_processed.pkl
Đầu ra  : /data/splits/{train,val,test}_idx.npy
           /data/splits/split_info.json
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/data/processed")
SPLITS_DIR    = os.environ.get("SPLITS_DIR", "/data/splits")
RANDOM_SEED   = int(os.environ.get("RANDOM_SEED", "42"))

TAB_PATH = os.path.join(PROCESSED_DIR, "tabular_processed.pkl")
os.makedirs(SPLITS_DIR, exist_ok=True)


def main():
    print(f"Đọc {TAB_PATH}")
    df = pd.read_pickle(TAB_PATH)

    y = df["target"].values
    indices = np.arange(len(df))

    print(f"Tổng mẫu: {len(indices)}")
    print(f"Malignant: {y.sum()} ({100*y.mean():.2f}%)")
    print(f"Benign   : {(1-y).sum()} ({100*(1-y).mean():.2f}%)")

    # Split: 80% trainval / 20% test (stratified)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.20,
        stratify=y, random_state=RANDOM_SEED
    )

    # Split trainval → 80% train / 20% val = 64% / 16% tổng thể
    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.20,
        stratify=y_trainval, random_state=RANDOM_SEED
    )

    np.save(os.path.join(SPLITS_DIR, "train_idx.npy"), idx_train)
    np.save(os.path.join(SPLITS_DIR, "val_idx.npy"),   idx_val)
    np.save(os.path.join(SPLITS_DIR, "test_idx.npy"),  idx_test)

    # Thống kê split
    def split_stats(name, idx):
        ys = y[idx]
        return {
            "total": len(idx),
            "malignant": int(ys.sum()),
            "benign": int((1-ys).sum()),
            "malignant_ratio": round(float(ys.mean()), 4)
        }

    info = {
        "random_seed": RANDOM_SEED,
        "train": split_stats("train", idx_train),
        "val":   split_stats("val",   idx_val),
        "test":  split_stats("test",  idx_test),
    }

    with open(os.path.join(SPLITS_DIR, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\nKết quả split:")
    for split_name, stats in info.items():
        if isinstance(stats, dict):
            print(f"  {split_name:6s}: {stats['total']:>7,} mẫu | "
                  f"Malignant {stats['malignant']:>6,} ({100*stats['malignant_ratio']:.2f}%)")

    print(f"\nLưu indices → {SPLITS_DIR}")


if __name__ == "__main__":
    main()
