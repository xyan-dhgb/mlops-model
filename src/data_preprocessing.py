"""
src/data_preprocessing.py
Core data preprocessing entry point for the MLOps pipeline.
Orchestrates image + tabular preprocessing, outputs artifacts for Feast/S3.
Called by: scripts/run_train.py and CI/CD pipeline
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import from Multimodal module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Multimodal.preprocessing.image_preprocessing import (
    preprocess_image,
    get_train_transforms,
    get_val_transforms,
    compute_class_weights,
    CLASS_NAMES,
)
from Multimodal.preprocessing.tabular_preprocessing import (
    clean_metadata,
    create_folds,
    MetadataPreprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Integrity: SHA-256 hashing
# SecMLOps: detect poisoned/tampered training data
# ─────────────────────────────────────────────
def hash_file_sha256(path: str) -> str:
    """Compute SHA-256 of a file. Used to audit training data integrity."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset_hashes(
    image_dir: str,
    expected_hashes: dict,
    sample_size: int = 100,
) -> bool:
    """
    Spot-check a random sample of images against known good hashes.
    Flags potential data poisoning (tampered images).
    Returns True if all sampled images match.
    """
    image_paths = list(Path(image_dir).glob("*.jpg"))
    if not image_paths:
        log.warning("No .jpg images found in %s", image_dir)
        return False

    np.random.shuffle(image_paths)
    sample = image_paths[:min(sample_size, len(image_paths))]
    mismatches = []

    for path in sample:
        name = path.stem
        if name in expected_hashes:
            actual = hash_file_sha256(str(path))
            if actual != expected_hashes[name]:
                mismatches.append(name)

    if mismatches:
        log.error("Hash mismatch for %d files: %s", len(mismatches), mismatches[:5])
        return False

    log.info("Hash validation passed (%d sampled files)", len(sample))
    return True


# ─────────────────────────────────────────────
# Data Validation
# ─────────────────────────────────────────────
def validate_metadata_csv(df: pd.DataFrame) -> dict:
    """
    Schema validation for ISIC metadata CSV.
    Returns a report dict: {valid: bool, issues: list}
    """
    issues = []
    required_cols = ["image_name", "age_approx", "sex",
                     "anatom_site_general_challenge", "diagnosis"]

    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    if "diagnosis" in df.columns:
        unknown_classes = set(df["diagnosis"].dropna().unique()) - set(CLASS_NAMES)
        if unknown_classes:
            issues.append(f"Unknown class labels: {unknown_classes}")

    if "age_approx" in df.columns:
        invalid_age = df[(df["age_approx"] < 0) | (df["age_approx"] > 120)].shape[0]
        if invalid_age > 0:
            issues.append(f"{invalid_age} rows with implausible age values")

    null_pct = df.isnull().mean() * 100
    high_null = null_pct[null_pct > 30]
    if not high_null.empty:
        issues.append(f"Columns with >30% nulls: {high_null.to_dict()}")

    report = {"valid": len(issues) == 0, "issues": issues, "shape": list(df.shape)}
    log.info("Validation: %s | Issues: %s", report["valid"], issues or "none")
    return report


# ─────────────────────────────────────────────
# Main Preprocessing Pipeline
# ─────────────────────────────────────────────
def run_preprocessing(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    n_folds: int = 5,
    seed: int = 42,
    hash_manifest: str = None,
) -> dict:
    """
    Full preprocessing pipeline:
    1. Validate CSV schema
    2. (Optional) validate image hashes
    3. Clean metadata
    4. Engineer features + create folds
    5. Fit MetadataPreprocessor on train split (fold 0 = val)
    6. Compute class weights
    7. Save artifacts to output_dir

    Returns: artifact paths dict
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load + validate
    log.info("Loading metadata from %s", csv_path)
    df_raw = pd.read_csv(csv_path)
    report = validate_metadata_csv(df_raw)
    if not report["valid"]:
        log.warning("Validation issues found: %s", report["issues"])

    # 2. Hash check
    if hash_manifest and Path(hash_manifest).exists():
        with open(hash_manifest) as f:
            expected = json.load(f)
        validate_dataset_hashes(image_dir, expected)

    # 3–4. Clean + fold
    df = clean_metadata(df_raw)
    df = create_folds(df, n_splits=n_folds, seed=seed)
    log.info("Dataset: %d samples, %d folds", len(df), n_folds)

    # 5. Fit preprocessor on train (fold != 0)
    train_df = df[df["fold"] != 0]
    preprocessor = MetadataPreprocessor()
    preprocessor.fit(train_df)
    pp_path = str(out / "metadata_preprocessor.pkl")
    preprocessor.save(pp_path)

    # 6. Class weights
    weights = compute_class_weights(df)
    weights_path = str(out / "class_weights.pt")
    torch.save(weights, weights_path)
    log.info("Class weights: %s", dict(zip(CLASS_NAMES, weights.tolist())))

    # 7. Save cleaned CSV + fold assignments
    folds_path = str(out / "metadata_with_folds.csv")
    df.to_csv(folds_path, index=False)

    # Save validation report
    report_path = str(out / "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    artifacts = {
        "metadata_with_folds": folds_path,
        "preprocessor": pp_path,
        "class_weights": weights_path,
        "validation_report": report_path,
    }
    log.info("Preprocessing complete. Artifacts: %s", artifacts)
    return artifacts


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing pipeline")
    parser.add_argument("--csv",        required=True,  help="Path to ISIC metadata CSV")
    parser.add_argument("--image-dir",  required=True,  help="Directory with .jpg images")
    parser.add_argument("--output-dir", required=True,  help="Output directory for artifacts")
    parser.add_argument("--n-folds",    default=5,  type=int)
    parser.add_argument("--seed",       default=42, type=int)
    parser.add_argument("--hash-manifest", default=None, help="JSON of expected SHA-256 hashes")
    args = parser.parse_args()

    run_preprocessing(
        csv_path=args.csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        seed=args.seed,
        hash_manifest=args.hash_manifest,
    )
