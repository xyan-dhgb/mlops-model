"""
train.py — Vòng lặp huấn luyện Multimodal Skin Cancer Detection (ISIC 2024)

Thay đổi so với ISIC 2019:
  - Task        : Binary (logit đơn, BCEWithLogitsLoss) thay vì 7-class softmax
  - Loss        : BinaryFocalLoss + pos_weight (~33 cho malignant ~3%)
  - Metric chính: pAUC (partial AUC tại TPR ≥ 0.80) theo đúng Kaggle ISIC 2024
  - Metric phụ  : AUC-ROC, Balanced Accuracy, F1-binary
  - DataLoader  : nhận hdf5_path thay vì image_dir
  - best_metric : val/pauc (thay vì val/auc_roc_macro)

Cách dùng:
    python scripts/run_train.py --config Multimodal/config/train_config.yaml
    python scripts/run_train.py --config Multimodal/config/train_config.yaml --fold 1
"""

import argparse
import time
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)
import mlflow
import mlflow.pytorch
import pandas as pd

from Multimodal.models.multimodal_model import MultimodalSkinClassifier, BinaryFocalLoss
from Multimodal.data_loader.dataloader import build_dataloaders
from Multimodal.preprocessing.tabular_preprocessing import (
    clean_metadata,
    compute_pos_weight,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config mặc định — ISIC 2024
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "experiment_name":     "multimodal_skin_cancer_isic2024",
    "run_name":            "efficientnet_b3_binary_fold0",
    "csv_path":            "Multimodal/data/raw/train-metadata.csv",
    "hdf5_path":           "Multimodal/data/raw/train-image.hdf5",
    "fold":                0,
    "num_epochs":          30,
    "batch_size":          32,
    "lr":                  1e-4,
    "weight_decay":        1e-4,
    "gamma_focal":         2.0,
    "focal_alpha":         0.25,
    "num_workers":         4,
    "device":              "cuda",
    "use_amp":             True,
    "save_dir":            "Multimodal/final",
    "best_metric":         "val/pauc",
    "pauc_min_tpr":        0.80,       # pAUC tại TPR ≥ 80% theo Kaggle ISIC 2024
    "grad_clip_norm":      1.0,
    "scheduler_t_max":     30,
    "mlflow_tracking_uri": "http://localhost:5000",
    "num_classes":         1,
    "metadata_input_dim":  9,
    "pretrained":          True,
    "freeze_bn":           False,
    "use_weighted_sampler": True,
    "image_size":          224,
    "apply_hair_removal":  False,
    "apply_color_constancy": True,
    "preprocessor_path":   None,
    "n_folds":             5,
    "seed":                42,
}


# ─────────────────────────────────────────────────────────────────────────────
# pAUC — metric chính của ISIC 2024
# ─────────────────────────────────────────────────────────────────────────────
def compute_pauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """
    Tính partial AUC (pAUC) tại vùng TPR ≥ min_tpr.
    Đây là metric chính thức của Kaggle ISIC 2024 Challenge.

    pAUC = diện tích dưới ROC curve trong khoảng FPR tương ứng với TPR ≥ min_tpr,
    chuẩn hóa về [0, 1] bằng cách chia cho chiều rộng vùng.

    Args:
        y_true   : nhãn binary (0/1)
        y_score  : xác suất malignant (sau sigmoid)
        min_tpr  : ngưỡng TPR tối thiểu (0.80 theo ISIC 2024)

    Returns:
        pAUC trong [0, 1]
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Tìm ngưỡng FPR tương ứng với TPR = min_tpr (nội suy)
    # Vùng tính pAUC: FPR thuộc [0, fpr_at_min_tpr]
    if tpr[0] >= min_tpr:
        # Toàn bộ curve nằm trong vùng tính
        fpr_at_min_tpr = fpr[0]
    else:
        idx = np.searchsorted(tpr, min_tpr)
        if idx >= len(tpr):
            return 0.0
        # Nội suy tuyến tính
        if idx > 0 and tpr[idx - 1] < min_tpr:
            slope = (fpr[idx] - fpr[idx - 1]) / max(tpr[idx] - tpr[idx - 1], 1e-9)
            fpr_at_min_tpr = fpr[idx - 1] + slope * (min_tpr - tpr[idx - 1])
        else:
            fpr_at_min_tpr = fpr[idx]

    # Lọc các điểm có TPR ≥ min_tpr
    mask = tpr >= min_tpr
    tpr_clip = np.concatenate([[min_tpr], tpr[mask]])
    fpr_clip = np.concatenate([[fpr_at_min_tpr], fpr[mask]])

    if len(fpr_clip) < 2:
        return 0.0

    # Tính AUC bằng hình thang, chuẩn hóa về [0, 1]
    width = fpr_clip[-1] - fpr_clip[0]
    if width <= 0:
        return 0.0

    pauc = np.trapz(tpr_clip, fpr_clip) / width
    return float(np.clip(pauc, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Train một epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
    grad_clip_norm: float = 1.0,
):
    """
    Huấn luyện một epoch cho binary task.
    Trả về (avg_loss, balanced_accuracy).
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, meta, labels in loader:
        imgs   = imgs.to(device)
        meta   = meta.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        if scaler:  # AMP
            with torch.cuda.amp.autocast():
                logits = model(imgs, meta)          # (B,) logit đơn
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs, meta)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()

        # Binary prediction: sigmoid > 0.5
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / len(loader)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    pauc_min_tpr: float = 0.80,
):
    """
    Đánh giá một epoch cho binary task.

    Trả về tuple:
        (avg_loss, bal_acc, f1, auc_roc, pauc, all_probs, all_labels)
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for imgs, meta, labels in loader:
        imgs   = imgs.to(device)
        meta   = meta.to(device)
        labels = labels.to(device).float()

        logits = model(imgs, meta)              # (B,) logit đơn
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()   # xác suất malignant
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().cpu().numpy().tolist())

    avg_loss   = total_loss / len(loader)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # F1 binary
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC-ROC toàn phần
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # pAUC (metric chính ISIC 2024)
    pauc = compute_pauc(all_labels, all_probs, min_tpr=pauc_min_tpr)

    return avg_loss, bal_acc, f1, auc, pauc, all_probs, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Vòng lặp huấn luyện chính
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params({
            k: v for k, v in cfg.items()
            if k not in ("csv_path", "hdf5_path", "save_dir", "preprocessor_path")
        })

        # ── DataLoader ────────────────────────────────────────────────────────
        train_loader, val_loader, preprocessor = build_dataloaders(
            csv_path=cfg["csv_path"],
            hdf5_path=cfg["hdf5_path"],
            fold=cfg.get("fold", 0),
            batch_size=cfg.get("batch_size", 32),
            num_workers=cfg.get("num_workers", 4),
            image_size=cfg.get("image_size", 224),
            use_weighted_sampler=cfg.get("use_weighted_sampler", True),
            apply_hair_removal=cfg.get("apply_hair_removal", False),
            apply_color_constancy=cfg.get("apply_color_constancy", True),
            preprocessor_path=cfg.get("preprocessor_path"),
            n_folds=cfg.get("n_folds", 5),
            seed=cfg.get("seed", 42),
            pin_memory=(device.type == "cuda"),
        )

        # ── pos_weight cho BinaryFocalLoss ────────────────────────────────────
        df_raw = pd.read_csv(cfg["csv_path"])
        df     = clean_metadata(df_raw)
        pos_weight = compute_pos_weight(df).to(device)
        print(f"pos_weight = {pos_weight.item():.2f}")

        # ── Model + Loss + Optimizer ──────────────────────────────────────────
        model = MultimodalSkinClassifier(
            num_classes=cfg.get("num_classes", 1),
            metadata_input_dim=cfg.get("metadata_input_dim", 9),
            pretrained=cfg.get("pretrained", True),
            freeze_bn=cfg.get("freeze_bn", False),
        ).to(device)

        criterion = BinaryFocalLoss(
            gamma=cfg.get("gamma_focal", 2.0),
            pos_weight=pos_weight,
            alpha=cfg.get("focal_alpha", 0.25),
        )
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("scheduler_t_max", cfg.get("num_epochs", 30)),
        )

        scaler = (
            torch.cuda.amp.GradScaler()
            if device.type == "cuda" and cfg.get("use_amp", True)
            else None
        )

        # ── Vòng lặp epoch ────────────────────────────────────────────────────
        best_metric_value = 0.0
        best_metric_key   = cfg.get("best_metric", "val/pauc")
        pauc_min_tpr      = cfg.get("pauc_min_tpr", 0.80)
        grad_clip_norm    = cfg.get("grad_clip_norm", 1.0)

        save_dir = Path(cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg["num_epochs"] + 1):
            t0 = time.time()

            train_loss, train_bal_acc = train_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, grad_clip_norm,
            )
            val_loss, val_bal_acc, val_f1, val_auc, val_pauc, _, _ = val_epoch(
                model, val_loader, criterion, device, pauc_min_tpr,
            )
            scheduler.step()
            elapsed = time.time() - t0

            # ── Ghi log MLflow ────────────────────────────────────────────────
            metrics = {
                "train/loss":              train_loss,
                "train/balanced_accuracy": train_bal_acc,
                "val/loss":                val_loss,
                "val/balanced_accuracy":   val_bal_acc,
                "val/f1_binary":           val_f1,
                "val/auc_roc":             val_auc,
                "val/pauc":                val_pauc,
                "lr":                      scheduler.get_last_lr()[0],
            }
            mlflow.log_metrics(metrics, step=epoch)

            print(
                f"Epoch {epoch:03d}/{cfg['num_epochs']} "
                f"| loss {train_loss:.4f}/{val_loss:.4f} "
                f"| bal_acc {train_bal_acc:.4f}/{val_bal_acc:.4f} "
                f"| AUC {val_auc:.4f} | pAUC {val_pauc:.4f} "
                f"| F1 {val_f1:.4f} | {elapsed:.1f}s"
            )

            # ── Lưu checkpoint tốt nhất ───────────────────────────────────────
            current_value = metrics.get(best_metric_key, val_pauc)
            if current_value > best_metric_value:
                best_metric_value = current_value
                ckpt_path = save_dir / f"best_model_fold{cfg['fold']}.pt"
                torch.save(model.state_dict(), str(ckpt_path))
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                print(f"  ✓ Best {best_metric_key} = {best_metric_value:.4f} — đã lưu checkpoint")

        # ── Đăng ký model lên MLflow Registry ────────────────────────────────
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="multimodal_skin_cancer_isic2024",
        )

        # ── Lưu preprocessor artifact ─────────────────────────────────────────
        pp_path = save_dir / "metadata_preprocessor.pkl"
        preprocessor.save(str(pp_path))
        mlflow.log_artifact(str(pp_path), artifact_path="artifacts")

        mlflow.log_metric(f"best_{best_metric_key.replace('/', '_')}", best_metric_value)
        print(f"\nHuấn luyện hoàn tất. Best {best_metric_key}: {best_metric_value:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISIC 2024 Multimodal Model")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    train(cfg)
