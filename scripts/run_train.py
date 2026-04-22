#!/usr/bin/env python3
"""
scripts/run_train.py
====================
Convenience wrapper – sets sensible defaults and invokes
MLflow_signature/train.py.

Typical usage inside Docker:
  python scripts/run_train.py

Override via env-vars:
  CSV_PATH, HDF5_PATH, IMAGE_DIR, OUTPUT_DIR, EXPERIMENT_NAME,
  PHASE1_EPOCHS, PHASE2_EPOCHS, BATCH_SIZE,
  MLFLOW_TRACKING_URI
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(ROOT, "MLflow_signature", "train.py")


def env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def main() -> None:
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--csv_path",       env("CSV_PATH",      "/data/train-metadata.csv"),
        "--hdf5_path",      env("HDF5_PATH",     "/data/train-image.hdf5"),
        "--image_dir",      env("IMAGE_DIR",     "/data/images"),
        "--output_dir",     env("OUTPUT_DIR",    "/models"),
        "--experiment",     env("EXPERIMENT_NAME", "isic2024_efficientnetb3"),
        "--phase1_epochs",  env("PHASE1_EPOCHS", "15"),
        "--phase2_epochs",  env("PHASE2_EPOCHS", "15"),
        "--batch_size",     env("BATCH_SIZE",    "32"),
        "--oversample_ratio", env("OVERSAMPLE_RATIO", "0.25"),
        "--n_benign",       env("N_BENIGN",      "4000"),
    ]

    if env("SKIP_EXTRACT", "false").lower() == "true":
        cmd.append("--skip_extract")
    if env("NO_PHASE2", "false").lower() == "true":
        cmd.append("--no_phase2")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
