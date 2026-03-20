"""
Image Preprocessing Pipeline — ISIC 2024 Skin Cancer Detection with 3D-TBP
Dataset : ISIC 2024 Challenge (~400k anh, binary: malignant=1 / benign=0)
Nguon anh: 3D Total Body Photography (TBP) crops — luu trong file HDF5
Backbone : EfficientNet-B3 (224x224)

Diem khac biet so voi ISIC 2019:
  - Anh luu trong image.hdf5, khong phai tung .jpg rieng
  - ID anh la isic_id, khong phai image_name
  - Nhan la binary (target: 0/1), khong phai 7-class
  - Mat can bang: malignant ~3% (nang hon melanoma ~11%)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import albumentations as A

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hang so
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES  = ["benign", "malignant"]
NUM_CLASSES  = 1          # sigmoid output (BCEWithLogitsLoss)
IMAGE_ID_COL = "isic_id"  # ISIC 2024: doi tu image_name


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 Image Loader — ISIC 2024 luu anh dang JPEG bytes ben trong HDF5
# ─────────────────────────────────────────────────────────────────────────────
class HDF5ImageStore:
    """
    Lazy-load anh tu file HDF5 cua ISIC 2024.

    Cau truc HDF5:
        /isic_id_1  ->  binary JPEG bytes
        /isic_id_2  ->  binary JPEG bytes

    Moi DataLoader worker mo handle rieng (an toan voi fork).
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = str(hdf5_path)
        self._handle: Optional[h5py.File] = None

    def _get_handle(self) -> h5py.File:
        if self._handle is None or not self._handle.id.valid:
            self._handle = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._handle

    def read(self, isic_id: str) -> np.ndarray:
        handle = self._get_handle()
        jpeg_bytes = handle[isic_id][()]
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Khong decode duoc anh: {isic_id}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __contains__(self, isic_id: str) -> bool:
        return isic_id in self._get_handle()

    def close(self):
        if self._handle and self._handle.id.valid:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()


def load_image_from_hdf5(hdf5_path: str, isic_id: str) -> np.ndarray:
    """Doc mot anh don le tu HDF5 (khong giu handle)."""
    with h5py.File(hdf5_path, "r") as f:
        jpeg_bytes = f[isic_id][()]
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Tien xu ly anh
# ─────────────────────────────────────────────────────────────────────────────
def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    DullRazor: morphological blackhat + INPAINT_TELEA.
    Tat mac dinh voi TBP (long it hon dermoscopy).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def shades_of_gray(image: np.ndarray, power: int = 6) -> np.ndarray:
    """
    Shades of Gray color constancy — chuan hoa sai lech anh sang.
    Van huu ich voi TBP vi anh chup tu nhieu thiet bi khac nhau.
    """
    image = image.astype(np.float32)
    norm = np.power(
        np.mean(np.power(image, power), axis=(0, 1), keepdims=True),
        1.0 / power,
    )
    image = image * (norm.mean() / (norm + 1e-6))
    return np.clip(image, 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    apply_hair_removal: bool = False,
    apply_color_constancy: bool = True,
) -> np.ndarray:
    """Ap dung tien xu ly len numpy array RGB, tra ve RGB uint8."""
    if apply_hair_removal:
        image = remove_hair(image)
    if apply_color_constancy:
        image = shades_of_gray(image)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """
    Augmentation manh cho ISIC 2024 — bu cho malignant ~3%.
    TBP crops co nhieu sai lech mau sac hon dermoscopy.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05, p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.3
        ),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size // 8,
            max_width=image_size // 8,
            p=0.2,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Deterministic val/test transforms — chi resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset — ISIC 2024
# ─────────────────────────────────────────────────────────────────────────────
class ISICDataset(Dataset):
    """
    ISIC 2024 multimodal dataset.
    Doc anh tu HDF5, metadata tu DataFrame da qua MetadataPreprocessor.
    Tra ve: (image_tensor, metadata_tensor, label) moi sample.

    label la torch.Tensor float32 scalar (0.0 hoac 1.0) cho BCEWithLogitsLoss.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hdf5_path: str,
        transforms: Optional[A.Compose] = None,
        apply_hair_removal: bool = False,
        apply_color_constancy: bool = True,
        metadata_dim: int = 9,
    ):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = str(hdf5_path)
        self.transforms = transforms
        self.apply_hair_removal = apply_hair_removal
        self.apply_color_constancy = apply_color_constancy
        self.metadata_dim = metadata_dim
        self._store: Optional[HDF5ImageStore] = None

    def _get_store(self) -> HDF5ImageStore:
        """Lazy init — an toan voi DataLoader fork."""
        if self._store is None:
            self._store = HDF5ImageStore(self.hdf5_path)
        return self._store

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        isic_id = str(row[IMAGE_ID_COL])

        # Doc anh tu HDF5
        try:
            img = self._get_store().read(isic_id)
        except (KeyError, ValueError) as e:
            log.warning("Khong doc duoc %s: %s — dung anh trang", isic_id, e)
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        img = preprocess_image(img, self.apply_hair_removal, self.apply_color_constancy)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        # Metadata (da duoc precompute boi MetadataPreprocessor)
        if "meta_features" in row.index and row["meta_features"] is not None:
            meta = torch.tensor(np.array(row["meta_features"]), dtype=torch.float32)
        else:
            meta = torch.zeros(self.metadata_dim, dtype=torch.float32)

        # Nhan binary float32 cho BCEWithLogitsLoss
        label = torch.tensor(
            float(row["target"]) if "target" in row.index else -1.0,
            dtype=torch.float32,
        )

        return img, meta, label


# ─────────────────────────────────────────────────────────────────────────────
# Tinh trong so mau cho WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────────
def compute_sample_weights(df: pd.DataFrame, target_col: str = "target") -> torch.Tensor:
    """
    Trong so moi mau theo inverse frequency cua nhan binary.
    malignant (1) nhan trong so cao hon benign (0).
    Returns: (N,) float tensor cho WeightedRandomSampler.
    """
    counts = df[target_col].value_counts()
    n = len(df)
    weight_map = {cls: n / (2 * cnt) for cls, cnt in counts.items()}
    weights = df[target_col].map(weight_map).values
    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weight(df: pd.DataFrame, target_col: str = "target") -> torch.Tensor:
    """
    pos_weight cho BCEWithLogitsLoss: n_negative / n_positive.
    Dung ket hop voi WeightedRandomSampler de xu ly mat can bang.
    """
    n_pos = (df[target_col] == 1).sum()
    n_neg = (df[target_col] == 0).sum()
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Image Preprocessing Module (ISIC 2024) ===")
    print(f"Target size      : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Task             : Binary — {CLASS_NAMES}")
    print(f"Image source     : HDF5 (image.hdf5)")
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    out = get_train_transforms()(image=dummy)["image"]
    print(f"Train transform  : {out.shape} | dtype={out.dtype}")
    out_v = get_val_transforms()(image=dummy)["image"]
    print(f"Val   transform  : {out_v.shape} | dtype={out_v.dtype}")
    print("OK")
