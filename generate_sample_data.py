#!/usr/bin/env python3
"""
scripts/generate_sample_data.py
Tạo dataset nhỏ (ảnh + CSV) cho CI integration test
Không cần download HAM10000 thật
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LABEL_MAP = {
    "akiec": "ACK", "bcc": "BCC", "mel": "MEL",
    "nv": "NEV",    "df": "SCC", "vasc": "SEK", "bkl": "SEK"
}


def generate(output_dir: str, samples_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    img_id = 0

    for dx_code in LABEL_MAP.keys():
        for i in range(samples_per_class):
            # Ảnh ngẫu nhiên kích thước khác nhau để test resize
            h = rng.integers(200, 500)
            w = rng.integers(200, 500)
            arr = rng.integers(40, 210, (h, w, 3), dtype=np.uint8)

            name = f"ISIC_{img_id:07d}"
            path = img_dir / f"{name}.jpg"
            Image.fromarray(arr).save(str(path))

            rows.append({
                "image_id":     name,
                "age":          rng.choice([*range(20, 80), None]),
                "sex":          rng.choice(["male", "female", "unknown"]),
                "localization": rng.choice(["back", "face", "trunk", "lower extremity"]),
                "dx":           dx_code,
                "dx_type":      rng.choice(["histo", "follow_up", "consensus"]),
            })
            img_id += 1

    df = pd.DataFrame(rows)
    csv_path = out / "metadata.csv"
    df.to_csv(str(csv_path), index=False)

    print(f"✅ Generated {len(df)} samples in {output_dir}")
    print(f"   Images: {img_dir}")
    print(f"   CSV:    {csv_path}")
    print(f"   Distribution:\n{df['dx'].value_counts().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",        default="/tmp/sample_data")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.samples_per_class, args.seed)
