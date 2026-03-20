"""
dataloader.py — Unified DataLoader cho ISIC 2024 Multimodal Pipeline
Ket hop ISICDataset (anh HDF5) voi MetadataPreprocessor (tabular 9-dim)
Dau ra moi batch: (image_tensor, metadata_tensor, label_tensor)

Thay doi so voi ISIC 2019:
  - Nguon anh: HDF5 thay vi thu muc .jpg
  - Cross-val : StratifiedGroupKFold(patient_id) tranh data leakage
  - Sampler   : WeightedRandomSampler voi trong so theo target binary
  - metadata_dim: 9 (cu: 5)
  - label     : torch.float32 scalar (cu: int)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from preprocessing.image_preprocessing import (
    ISICDataset,
    compute_sample_weights,
    get_train_transforms,
    get_val_transforms,
    IMAGE_SIZE,
    IMAGE_ID_COL,
)
from preprocessing.tabular_preprocessing import (
    MetadataPreprocessor,
    clean_metadata,
    compute_pos_weight,
    create_folds,
    FEATURE_DIM,
    TARGET_COL,
    GROUP_COL,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ham chinh — build tu tham so tuong minh
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    csv_path: str,
    hdf5_path: str,
    fold: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = IMAGE_SIZE,
    use_weighted_sampler: bool = True,
    apply_hair_removal: bool = False,
    apply_color_constancy: bool = True,
    preprocessor_path: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, MetadataPreprocessor, torch.Tensor]:
    """
    Xay dung train/val DataLoader cho mot fold cu the.

    Pipeline:
        CSV -> clean_metadata -> create_folds (StratifiedGroupKFold)
        -> train_df / val_df
        -> MetadataPreprocessor.attach_to_df() (precompute meta_features)
        -> ISICDataset (doc anh tu HDF5)
        -> DataLoader (train: WeightedRandomSampler | val: tuan tu)

    Returns:
        train_loader, val_loader, preprocessor, pos_weight
        (pos_weight dung truc tiep voi BinaryFocalLoss)
    """
    # 1. Tai va lam sach CSV
    log.info("Dang tai CSV: %s", csv_path)
    df_raw = pd.read_csv(csv_path)
    df = clean_metadata(df_raw)
    log.info("clean_metadata: %d mau hop le / %d tong", len(df), len(df_raw))

    # 2. Tao fold neu chua co
    if "fold" not in df.columns:
        df = create_folds(df, n_splits=n_folds, seed=seed)

    # 3. Phan chia train / val
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df   = df[df["fold"] == fold].reset_index(drop=True)
    log.info("[Fold %d] Train: %d | Val: %d", fold, len(train_df), len(val_df))
    _log_target_dist(train_df, "Train")
    _log_target_dist(val_df,   "Val")

    # 4. Fit hoac load MetadataPreprocessor
    preprocessor = _get_preprocessor(preprocessor_path, train_df)

    # 5. Precompute meta_features va attach vao DataFrame
    train_df = preprocessor.attach_to_df(train_df)
    val_df   = preprocessor.attach_to_df(val_df)

    # 6. Tinh pos_weight cho BinaryFocalLoss
    pos_weight = compute_pos_weight(train_df)
    log.info("pos_weight = %.2f", pos_weight.item())

    # 7. Khoi tao Dataset
    train_dataset = ISICDataset(
        df=train_df,
        hdf5_path=hdf5_path,
        transforms=get_train_transforms(image_size),
        apply_hair_removal=apply_hair_removal,
        apply_color_constancy=apply_color_constancy,
        metadata_dim=FEATURE_DIM,
    )
    val_dataset = ISICDataset(
        df=val_df,
        hdf5_path=hdf5_path,
        transforms=get_val_transforms(image_size),
        apply_hair_removal=apply_hair_removal,
        apply_color_constancy=apply_color_constancy,
        metadata_dim=FEATURE_DIM,
    )

    # 8. WeightedRandomSampler cho train
    sampler = None
    if use_weighted_sampler:
        sample_weights = compute_sample_weights(train_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        log.info("WeightedRandomSampler: bat (malignant oversample)")
    else:
        log.info("WeightedRandomSampler: tat")

    # 9. DataLoader
    _persistent = persistent_workers and (num_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    log.info(
        "DataLoader san sang | batch=%d | workers=%d | "
        "train_batches=%d | val_batches=%d",
        batch_size, num_workers, len(train_loader), len(val_loader),
    )
    return train_loader, val_loader, preprocessor, pos_weight


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper tien loi — build tu dict config (train_config.yaml)
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders_from_config(
    cfg: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, MetadataPreprocessor, torch.Tensor]:
    """
    Nhan dict config tu train_config.yaml va goi build_dataloaders().

    Vi du:
        import yaml
        with open("Multimodal/config/train_config.yaml") as f:
            cfg = yaml.safe_load(f)
        train_loader, val_loader, pp, pos_w = build_dataloaders_from_config(cfg)
    """
    pin_memory = (cfg.get("device", "cuda") == "cuda")
    return build_dataloaders(
        csv_path=cfg["csv_path"],
        hdf5_path=cfg["hdf5_path"],
        fold=cfg.get("fold", 0),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 4),
        image_size=cfg.get("image_size", IMAGE_SIZE),
        use_weighted_sampler=cfg.get("use_weighted_sampler", True),
        apply_hair_removal=cfg.get("apply_hair_removal", False),
        apply_color_constancy=cfg.get("apply_color_constancy", True),
        preprocessor_path=cfg.get("preprocessor_path"),
        n_folds=cfg.get("n_folds", 5),
        seed=cfg.get("seed", 42),
        pin_memory=pin_memory,
        persistent_workers=cfg.get("num_workers", 4) > 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader cho inference / XAI / adversarial eval
# ─────────────────────────────────────────────────────────────────────────────
def build_inference_loader(
    df: pd.DataFrame,
    hdf5_path: str,
    preprocessor: MetadataPreprocessor,
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = IMAGE_SIZE,
    apply_hair_removal: bool = False,
    apply_color_constancy: bool = True,
) -> DataLoader:
    """
    DataLoader cho inference — label la -1.0 (khong co nhan that).
    Dung cho: XRAI/SHAP sau huan luyen, KServe batch request,
              adversarial robustness evaluation.
    """
    df_with_meta = preprocessor.attach_to_df(df)
    # Them cot target gia de ISICDataset khong bao loi
    if TARGET_COL not in df_with_meta.columns:
        df_with_meta[TARGET_COL] = -1

    dataset = ISICDataset(
        df=df_with_meta,
        hdf5_path=hdf5_path,
        transforms=get_val_transforms(image_size),
        apply_hair_removal=apply_hair_removal,
        apply_color_constancy=apply_color_constancy,
        metadata_dim=FEATURE_DIM,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ham ho tro noi bo
# ─────────────────────────────────────────────────────────────────────────────
def _get_preprocessor(
    preprocessor_path: Optional[str],
    train_df: pd.DataFrame,
) -> MetadataPreprocessor:
    if preprocessor_path:
        path = Path(preprocessor_path)
        if path.exists():
            log.info("Load MetadataPreprocessor: %s", preprocessor_path)
            return MetadataPreprocessor.load(str(path))
        log.warning("'%s' khong ton tai -> fit moi", preprocessor_path)

    log.info("Fit MetadataPreprocessor tren %d mau train", len(train_df))
    pp = MetadataPreprocessor()
    pp.fit(train_df)
    return pp


def _log_target_dist(df: pd.DataFrame, split_name: str) -> None:
    """Ghi log phan phoi nhan binary."""
    if TARGET_COL not in df.columns:
        return
    n_pos = (df[TARGET_COL] == 1).sum()
    n_neg = (df[TARGET_COL] == 0).sum()
    total = len(df)
    log.info(
        "[%s] benign=%d(%.1f%%) | malignant=%d(%.1f%%)",
        split_name,
        n_neg, 100 * n_neg / total,
        n_pos, 100 * n_pos / total,
    )


def get_loader_stats(loader: DataLoader, name: str = "Loader") -> Dict[str, Any]:
    """Tra ve dict thong ke co ban cua DataLoader."""
    stats = {
        "name":        name,
        "n_samples":   len(loader.dataset),
        "n_batches":   len(loader),
        "batch_size":  loader.batch_size,
        "num_workers": loader.num_workers,
        "drop_last":   loader.drop_last,
        "sampler":     loader.sampler.__class__.__name__,
    }
    log.info(
        "[%s] %d mau | %d batch | sampler=%s",
        name, stats["n_samples"], stats["n_batches"], stats["sampler"],
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config_path = Path(__file__).parent.parent / "config" / "train_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Config: fold={cfg['fold']} | batch={cfg['batch_size']} | hdf5={cfg.get('hdf5_path','N/A')}")
    print("DataLoader module OK — can CSV va HDF5 that de chay day du.")
