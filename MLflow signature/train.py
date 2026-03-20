"""
Training Script — Multimodal Skin Cancer Detection
MLflow experiment tracking: loss, AUC-ROC, balanced accuracy, F1 per class
Run: python training/train.py --config config/train_config.yaml
"""

import argparse
import yaml
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    f1_score, classification_report,
)
import mlflow
import mlflow.pytorch

from models.multimodal_model import MultimodalSkinClassifier, FocalLoss
from data_loader.dataloader import build_dataloaders
from preprocessing.tabular_preprocessing import compute_class_weights, clean_metadata, create_folds
import pandas as pd


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "experiment_name": "multimodal_skin_cancer",
    "run_name": "efficientnet_b3_focal_fold0",
    "csv_path": "data/raw/ISIC_2019_Training_Metadata.csv",
    "image_dir": "data/raw/ISIC_2019_Training_Input",
    "fold": 0,
    "num_epochs": 30,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "gamma_focal": 2.0,
    "num_workers": 4,
    "device": "cuda",
    "save_dir": "final",
    "mlflow_tracking_uri": "http://mlflow-service:5000",
}


# ─────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, meta, labels in loader:
        imgs   = imgs.to(device)
        meta   = meta.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler:  # AMP
            with torch.cuda.amp.autocast():
                logits = model(imgs, meta)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs, meta)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


# ─────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion, device, num_classes=7):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []

    for imgs, meta, labels in loader:
        imgs   = imgs.to(device)
        meta   = meta.to(device)
        labels = labels.to(device)

        logits = model(imgs, meta)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # AUC-ROC macro (one-vs-rest)
    try:
        auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr",
            average="macro", labels=list(range(num_classes))
        )
    except ValueError:
        auc = 0.0

    # Per-class F1
    f1_per_class = f1_score(all_labels, all_preds, average=None,
                            labels=list(range(num_classes)), zero_division=0)

    return avg_loss, bal_acc, f1_macro, auc, f1_per_class


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────
def train(cfg: dict):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # MLflow setup
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params({k: v for k, v in cfg.items()
                           if k not in ("csv_path", "image_dir", "save_dir")})

        # Data
        train_loader, val_loader, preprocessor = build_dataloaders(
            csv_path=cfg["csv_path"],
            image_dir=cfg["image_dir"],
            fold=cfg["fold"],
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
        )

        # Class weights
        df = pd.read_csv(cfg["csv_path"])
        from preprocessing.tabular_preprocessing import clean_metadata
        df = clean_metadata(df)
        class_weights = compute_class_weights(df).to(device)

        # Model
        model = MultimodalSkinClassifier(num_classes=7, pretrained=True).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=cfg["gamma_focal"])
        optimizer = AdamW(model.parameters(),
                          lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])

        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        best_auc = 0.0
        save_dir = Path(cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg["num_epochs"] + 1):
            t0 = time.time()
            train_loss, train_bal_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_bal_acc, val_f1, val_auc, val_f1_per_class = val_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step()
            elapsed = time.time() - t0

            # MLflow logging
            metrics = {
                "train/loss": train_loss,
                "train/balanced_accuracy": train_bal_acc,
                "val/loss": val_loss,
                "val/balanced_accuracy": val_bal_acc,
                "val/f1_macro": val_f1,
                "val/auc_roc_macro": val_auc,
                "lr": scheduler.get_last_lr()[0],
            }
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                metrics[f"val/f1_{cls_name}"] = float(val_f1_per_class[cls_idx])

            mlflow.log_metrics(metrics, step=epoch)

            print(
                f"Epoch {epoch:03d}/{cfg['num_epochs']} "
                f"| loss {train_loss:.4f}/{val_loss:.4f} "
                f"| bal_acc {train_bal_acc:.4f}/{val_bal_acc:.4f} "
                f"| AUC {val_auc:.4f} | F1 {val_f1:.4f} "
                f"| {elapsed:.1f}s"
            )

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                ckpt_path = save_dir / f"best_model_fold{cfg['fold']}.pt"
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                print(f"  ✓ Best AUC {best_auc:.4f} — checkpoint saved")

        # Log final model to MLflow registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="multimodal_skin_cancer_v1",
        )
        # Log preprocessor artifact
        pp_path = save_dir / "metadata_preprocessor.pkl"
        preprocessor.save(str(pp_path))
        mlflow.log_artifact(str(pp_path), artifact_path="artifacts")

        mlflow.log_metric("best_val_auc_roc", best_auc)
        print(f"\nTraining complete. Best AUC: {best_auc:.4f}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    train(cfg)
