"""
train.py  (MLflow Signature)
=============================
Entry-point for the full ISIC 2024 two-phase training run with MLflow tracking.

Usage
-----
python MLflow_signature/train.py \
    --csv_path      data/train-metadata.csv \
    --hdf5_path     data/train-image.hdf5 \
    --image_dir     data/images \
    --output_dir    models \
    --experiment    isic2024_efficientnetb3 \
    --phase1_epochs 15 \
    --phase2_epochs 15 \
    --batch_size    32

Environment
-----------
MLFLOW_TRACKING_URI : URI of the MLflow tracking server (default: ./mlruns)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mlflow
import mlflow.keras
import numpy as np

from data_preprocessing import (
    build_balanced_selected_ids,
    extract_images_from_hdf5,
    prepare_multimodal_data,
    stratified_split,
    oversample_malignant,
)
from model import (
    IMAGE_SHAPE,
    build_multimodal_model,
    compute_pauc,
    evaluate_model,
    find_optimal_threshold,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ISIC 2024 Multimodal Training")
    p.add_argument("--csv_path",       required=True)
    p.add_argument("--hdf5_path",      required=True)
    p.add_argument("--image_dir",      required=True)
    p.add_argument("--output_dir",     default="models")
    p.add_argument("--experiment",     default="isic2024_efficientnetb3")
    p.add_argument("--run_name",       default=None)
    p.add_argument("--phase1_epochs",  type=int, default=15)
    p.add_argument("--phase2_epochs",  type=int, default=15)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--n_benign",       type=int, default=4000,
                   help="Benign images to sample (all malignant always kept)")
    p.add_argument("--oversample_ratio", type=float, default=0.25)
    p.add_argument("--no_phase2",      action="store_true")
    p.add_argument("--skip_extract",   action="store_true",
                   help="Skip HDF5 extraction if images already exist")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.image_dir,  exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        # ── Log hyper-parameters ────────────────────────────────────────
        params = vars(args)
        mlflow.log_params(params)
        logger.info("MLflow Run ID: %s", run.info.run_id)

        # ── 1. HDF5 Extraction ──────────────────────────────────────────
        if not args.skip_extract:
            logger.info("Step 1/5: Extracting images from HDF5 …")
            selected_ids = build_balanced_selected_ids(
                args.csv_path, n_benign=args.n_benign
            )
            extract_images_from_hdf5(args.hdf5_path, args.image_dir, selected_ids)
        else:
            logger.info("Step 1/5: Skipping HDF5 extraction (--skip_extract).")

        # ── 2. Data Preparation ─────────────────────────────────────────
        logger.info("Step 2/5: Preparing multimodal data …")
        import pandas as pd
        df = pd.read_csv(args.csv_path)
        available_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(".jpg")
        }
        df = df[df["isic_id"].isin(available_ids)].reset_index(drop=True)
        logger.info("DataFrame filtered: %d rows (matched images)", len(df))

        X_tab, X_img, y, feature_cols, label_encoders, scaler, imputer = prepare_multimodal_data(
            df, args.image_dir, is_training=True
        )

        mlflow.log_metric("n_samples",   len(y))
        mlflow.log_metric("n_malignant", int(np.sum(y == 1)))
        mlflow.log_metric("n_benign",    int(np.sum(y == 0)))
        mlflow.log_metric("n_features",  X_tab.shape[1])

        # ── 3. Stratified Split ─────────────────────────────────────────
        logger.info("Step 3/5: Stratified split (64/16/20) …")
        splits = stratified_split(X_tab, X_img, y)
        X_tab_tr, X_img_tr, y_tr = splits["train"]
        X_tab_vl, X_img_vl, y_vl = splits["val"]
        X_tab_te, X_img_te, y_te = splits["test"]

        logger.info(
            "Train=%d | Val=%d | Test=%d",
            len(y_tr), len(y_vl), len(y_te),
        )

        # ── 4. Build & Train Model ──────────────────────────────────────
        logger.info("Step 4/5: Building EfficientNetB3 multimodal model …")
        tabular_shape = (X_tab_tr.shape[1],)
        model, backbone = build_multimodal_model(
            tabular_shape=tabular_shape,
            image_shape=IMAGE_SHAPE,
            freeze_backbone=True,
        )
        model.summary(print_fn=logger.info)

        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        history1, history2 = train_model(
            model, backbone,
            X_tab_tr, X_img_tr, y_tr,
            X_tab_vl, X_img_vl, y_vl,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            batch_size=args.batch_size,
            run_phase2=not args.no_phase2,
            checkpoint_dir=checkpoint_dir,
            oversample_ratio=args.oversample_ratio,
        )

        # Log training curves
        for epoch, vals in enumerate(zip(
            history1.history.get("val_auc", []),
            history1.history.get("val_recall", []),
            history1.history.get("val_loss", []),
        )):
            mlflow.log_metrics(
                {"phase1_val_auc": vals[0], "phase1_val_recall": vals[1], "phase1_val_loss": vals[2]},
                step=epoch,
            )

        if history2:
            offset = len(history1.history.get("val_auc", []))
            for i, vals in enumerate(zip(
                history2.history.get("val_auc", []),
                history2.history.get("val_recall", []),
                history2.history.get("val_loss", []),
            )):
                mlflow.log_metrics(
                    {"phase2_val_auc": vals[0], "phase2_val_recall": vals[1], "phase2_val_loss": vals[2]},
                    step=offset + i,
                )

        # ── 5. Evaluation ───────────────────────────────────────────────
        logger.info("Step 5/5: Evaluating on test set …")
        metrics = evaluate_model(
            model, X_tab_te, X_img_te, y_te,
            auto_tune_threshold=True,
        )

        mlflow.log_metrics(metrics)
        logger.info("Test metrics: %s", json.dumps(metrics, indent=2))

        # ── Save artefacts ───────────────────────────────────────────────
        model_path = os.path.join(args.output_dir, "multimodal_model.keras")
        model.save(model_path)
        mlflow.keras.log_model(model, artifact_path="model")

        # Save preprocessing objects
        import pickle
        preproc = {
            "feature_cols": feature_cols,
            "label_encoders": label_encoders,
            "scaler": scaler,
            "imputer": imputer,
            "threshold": metrics["threshold"],
        }
        preproc_path = os.path.join(args.output_dir, "preprocessors.pkl")
        with open(preproc_path, "wb") as f:
            pickle.dump(preproc, f)
        mlflow.log_artifact(preproc_path, artifact_path="preprocessors")

        # MLflow model signature
        import mlflow.models.signature as sig_utils
        input_schema  = mlflow.types.Schema([
            mlflow.types.TensorSpec(np.dtype("float32"), (-1, 224, 224, 3), "image_input"),
            mlflow.types.TensorSpec(np.dtype("float32"), (-1, X_tab_tr.shape[1]), "tabular_input"),
        ])
        output_schema = mlflow.types.Schema([
            mlflow.types.TensorSpec(np.dtype("float32"), (-1, 1), "probability_malignant"),
        ])
        signature = mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.keras.log_model(model, artifact_path="model_with_signature", signature=signature)

        logger.info("Training complete. Run ID: %s", run.info.run_id)
        logger.info("AUC=%.4f | pAUC=%.4f | F1=%.4f | threshold=%.2f",
                    metrics["auc_roc"], metrics["pauc_tpr80"],
                    metrics["f1_malignant"], metrics["threshold"])


if __name__ == "__main__":
    main()
