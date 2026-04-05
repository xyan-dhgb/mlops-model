"""
MLflow-integrated training entry point for ISIC 2024 multimodal model.
Logs params, metrics, model signature, and artifacts.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Multimodal.preprocessing.image_preprocessing import extract_images_from_hdf5
from Multimodal.preprocessing.tabular_preprocessing import (
    preprocess_csv_data, save_preprocessor, fit_encoders,
    encode_categorical, get_tabular_columns, scale_features,
    CATEGORICAL_COLS, ID_COL,
)
from Multimodal.data_loader.dataloader import build_dataloaders
from Multimodal.models.multimodal_model import build_multimodal_model
from Multimodal.training.train import train_model, evaluate_model, plot_training_history


def load_config(path: str = "Multimodal/config/train_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # ── MLflow setup ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["output"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["output"]["mlflow_experiment"])

    with mlflow.start_run():
        # ── Log config params ─────────────────────────────────────────────
        mlflow.log_params({
            "epochs":          cfg["training"]["epochs"],
            "batch_size":      cfg["training"]["batch_size"],
            "learning_rate":   cfg["training"]["learning_rate"],
            "image_size":      str(cfg["preprocessing"]["image_size"]),
            "test_size":       cfg["split"]["test_size"],
            "max_images":      cfg["data"].get("max_images", "all"),
            "apply_clahe":     cfg["preprocessing"]["apply_clahe"],
            "use_class_weights": cfg["training"]["use_class_weights"],
        })

        # ── Extract images from HDF5 if needed ───────────────────────────
        image_dir = cfg["data"]["image_dir"]
        if not os.path.exists(image_dir) or not os.listdir(image_dir):
            extract_images_from_hdf5(
                cfg["data"]["hdf5_path"],
                image_dir,
                max_images=cfg["data"].get("max_images"),
            )

        # ── Load and preprocess CSV ───────────────────────────────────────
        df_raw = pd.read_csv(cfg["data"]["csv_path"])

        # Keep only rows with extracted images
        available_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg")
        }
        df_raw = df_raw[df_raw[ID_COL].isin(available_ids)].reset_index(drop=True)
        print(f"Filtered DataFrame: {len(df_raw)} rows")

        df, preprocess_report = preprocess_csv_data(df_raw)

        # ── Build dataloaders ─────────────────────────────────────────────
        img_h, img_w = cfg["preprocessing"]["image_size"]
        splits = build_dataloaders(
            df, image_dir,
            target_size=(img_h, img_w),
            test_size=cfg["split"]["test_size"],
            val_size=cfg["split"]["val_size"],
            random_state=cfg["split"]["random_state"],
        )

        train, val, test = splits["train"], splits["val"], splits["test"]

        # Save preprocessor
        preprocessor = {
            "encoders":     splits["encoders"],
            "tabular_cols": splits["tabular_cols"],
        }
        save_preprocessor(preprocessor, cfg["output"]["preprocessor_path"])

        # ── Build model ───────────────────────────────────────────────────
        tabular_shape = (len(splits["tabular_cols"]),)
        image_shape   = tuple(cfg["model"]["image_shape"])
        num_classes   = cfg["model"]["num_classes"]

        model = build_multimodal_model(tabular_shape, image_shape, num_classes)
        model.summary()

        # ── Train ─────────────────────────────────────────────────────────
        history = train_model(
            model,
            train["X_tab"], train["X_img"], train["y"],
            val["X_tab"],   val["X_img"],   val["y"],
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            checkpoint_path=cfg["output"]["checkpoint_path"],
        )

        plot_training_history(history)

        # ── Evaluate ──────────────────────────────────────────────────────
        acc, auc_score = evaluate_model(
            model, test["X_tab"], test["X_img"], test["y"]
        )

        mlflow.log_metrics({"test_accuracy": acc, "test_auc": auc_score})

        # ── Log model with MLflow signature ───────────────────────────────
        import mlflow.pyfunc

        tab_sample = train["X_tab"][:1]
        img_sample = train["X_img"][:1]

        signature = mlflow.models.infer_signature(
            model_input={
                "image_input":   img_sample,
                "tabular_input": tab_sample,
            },
            model_output=model.predict(
                {"image_input": img_sample, "tabular_input": tab_sample}
            ),
        )

        mlflow.keras.log_model(
            model,
            artifact_path="multimodal_isic2024",
            signature=signature,
        )

        print(f"\n✅ MLflow run complete — Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")


if __name__ == "__main__":
    main()
