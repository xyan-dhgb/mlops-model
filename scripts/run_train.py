"""
scripts/run_train.py
CLI entry point: runs the full ISIC 2024 multimodal training pipeline.

Usage:
    python scripts/run_train.py [--config Multimodal/config/train_config.yaml]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ISIC 2024 multimodal skin-lesion classifier"
    )
    parser.add_argument(
        "--config",
        default="Multimodal/config/train_config.yaml",
        help="Path to YAML training config (default: Multimodal/config/train_config.yaml)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Run training without MLflow logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_mlflow:
        # Direct training without MLflow
        import yaml
        import numpy as np
        import pandas as pd

        from Multimodal.preprocessing.image_preprocessing import extract_images_from_hdf5
        from Multimodal.preprocessing.tabular_preprocessing import (
            preprocess_csv_data, save_preprocessor, ID_COL
        )
        from Multimodal.data_loader.dataloader import build_dataloaders
        from Multimodal.models.multimodal_model import build_multimodal_model
        from Multimodal.training.train import (
            train_model, evaluate_model, plot_training_history
        )

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # Extract images if needed
        image_dir = cfg["data"]["image_dir"]
        if not os.path.exists(image_dir) or not os.listdir(image_dir):
            extract_images_from_hdf5(
                cfg["data"]["hdf5_path"],
                image_dir,
                max_images=cfg["data"].get("max_images"),
            )

        df_raw = pd.read_csv(cfg["data"]["csv_path"])
        available_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir) if f.lower().endswith(".jpg")
        }
        df_raw = df_raw[df_raw[ID_COL].isin(available_ids)].reset_index(drop=True)

        df, _ = preprocess_csv_data(df_raw)

        img_h, img_w = cfg["preprocessing"]["image_size"]
        splits = build_dataloaders(
            df, image_dir, target_size=(img_h, img_w),
            test_size=cfg["split"]["test_size"],
            val_size=cfg["split"]["val_size"],
            random_state=cfg["split"]["random_state"],
        )

        train, val, test = splits["train"], splits["val"], splits["test"]
        save_preprocessor(
            {"encoders": splits["encoders"], "tabular_cols": splits["tabular_cols"]},
            cfg["output"]["preprocessor_path"],
        )

        model = build_multimodal_model(
            tabular_shape=(len(splits["tabular_cols"]),),
            image_shape=tuple(cfg["model"]["image_shape"]),
            num_classes=cfg["model"]["num_classes"],
        )

        history = train_model(
            model,
            train["X_tab"], train["X_img"], train["y"],
            val["X_tab"],   val["X_img"],   val["y"],
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            checkpoint_path=cfg["output"]["checkpoint_path"],
        )

        plot_training_history(history)
        acc, auc = evaluate_model(model, test["X_tab"], test["X_img"], test["y"])
        print(f"\n✅ Done — Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    else:
        # Run via MLflow-integrated train.py
        from MLflow_signature.train import main as mlflow_main
        mlflow_main()


if __name__ == "__main__":
    main()
