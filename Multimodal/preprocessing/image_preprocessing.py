"""
preprocessing/image_preprocessing.py
=====================================
Image loading, preprocessing, and augmentation for ISIC 2024.

Functions
---------
load_image            : Load an image file → np.ndarray (H, W, 3) float32
preprocess_image      : CLAHE + Gaussian + contrast enhancement
augment_image         : Random geometric / photometric augmentation
oversample_malignant  : Oversample minority class with strong augmentation
extract_images_from_hdf5 : Dump ISIC HDF5 to individual JPEG files
"""

import os
import random
import numpy as np
import cv2
import h5py
from PIL import Image, ImageEnhance


# ── LOAD ─────────────────────────────────────────────────────────────────────

def load_image(image_path: str,
               target_size: tuple = (224, 224),
               color_mode: str = "rgb") -> np.ndarray | None:
    """
    Load an image from disk and resize to target_size.

    Returns
    -------
    np.ndarray shape (H, W, 3) dtype float32, or None on failure.
    """
    try:
        img = Image.open(image_path)
        if color_mode == "rgb":
            img = img.convert("RGB")
        elif color_mode == "gray":
            img = img.convert("L")
        img = img.resize(target_size, Image.LANCZOS)
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"[load_image] ERROR loading {image_path}: {e}")
        return None


# ── PREPROCESS ───────────────────────────────────────────────────────────────

def preprocess_image(img_array: np.ndarray,
                     apply_clahe: bool = True,
                     apply_gaussian: bool = True,
                     enhance_contrast: float = 1.2) -> np.ndarray | None:
    """
    Apply clinical preprocessing to a skin-lesion image.

    Steps (all optional via config):
      1. CLAHE per LAB channel
      2. Gaussian blur (mild denoising)
      3. Contrast enhancement via PIL

    Parameters
    ----------
    img_array : float32 np.ndarray (H, W, 3), values 0–255
    apply_clahe : apply Contrast Limited Adaptive Histogram Equalization
    apply_gaussian : apply mild Gaussian smoothing
    enhance_contrast : PIL contrast multiplier (1.0 = no change)

    Returns
    -------
    np.ndarray (H, W, 3) float32 or None on failure.
    """
    try:
        img = img_array.astype(np.uint8)

        if apply_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if apply_gaussian:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        if enhance_contrast != 1.0:
            pil_img = Image.fromarray(img)
            pil_img = ImageEnhance.Contrast(pil_img).enhance(enhance_contrast)
            img = np.array(pil_img)

        return img.astype(np.float32)

    except Exception as e:
        print(f"[preprocess_image] ERROR: {e}")
        return None


# ── AUGMENTATION ─────────────────────────────────────────────────────────────

def augment_image(img_array: np.ndarray,
                  rotation_range: int = 15,
                  brightness_range: tuple = (0.8, 1.2),
                  zoom_range: float = 0.1,
                  horizontal_flip: bool = True,
                  vertical_flip: bool = False) -> np.ndarray:
    """
    Apply random augmentation to a single image.

    Parameters
    ----------
    img_array : float32 np.ndarray (H, W, 3)

    Returns
    -------
    Augmented np.ndarray (H, W, 3) float32.
    """
    img = img_array.astype(np.uint8)
    h, w = img.shape[:2]

    # Random rotation
    if rotation_range > 0:
        angle = random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_REFLECT_101)

    # Random zoom (crop + resize)
    if zoom_range > 0:
        factor = random.uniform(1.0, 1.0 + zoom_range)
        new_h, new_w = int(h * factor), int(w * factor)
        img_resized = cv2.resize(img, (new_w, new_h))
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        img = img_resized[start_y:start_y + h, start_x:start_x + w]

    # Random flips
    if horizontal_flip and random.random() > 0.5:
        img = cv2.flip(img, 1)
    if vertical_flip and random.random() > 0.5:
        img = cv2.flip(img, 0)

    # Brightness
    if brightness_range:
        factor = random.uniform(*brightness_range)
        pil_img = Image.fromarray(img)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)
        img = np.array(pil_img)

    return img.astype(np.float32)


# ── OVERSAMPLING ─────────────────────────────────────────────────────────────

def oversample_malignant(X_img: np.ndarray,
                         X_tab: np.ndarray,
                         y: np.ndarray,
                         target_ratio: float = 0.25,
                         strong_aug: bool = True) -> tuple:
    """
    Oversample the Malignant (minority) class using augmentation.

    Parameters
    ----------
    X_img        : (N, H, W, 3) image array
    X_tab        : (N, F) tabular feature array
    y            : (N,)  binary labels  0 = Benign, 1 = Malignant
    target_ratio : desired fraction of Malignant samples in the result
    strong_aug   : apply aggressive augmentation to synthetic samples

    Returns
    -------
    X_img_out, X_tab_out, y_out — shuffled together
    """
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    n_needed = int(target_ratio * n_neg / (1 - target_ratio)) - n_pos

    if n_needed <= 0:
        print("[oversample] No oversampling needed.")
        return X_img, X_tab, y

    print(f"[oversample] Generating {n_needed} synthetic Malignant samples "
          f"(current ratio: {n_pos / (n_neg + n_pos):.3f} → target: {target_ratio:.2f})")

    pos_idx = np.where(y == 1)[0]
    aug_imgs, aug_tabs, aug_labels = [], [], []

    for i in range(n_needed):
        src = pos_idx[i % len(pos_idx)]
        img = X_img[src].copy()

        if strong_aug:
            img = augment_image(img,
                                rotation_range=25,
                                brightness_range=(0.7, 1.3),
                                zoom_range=0.15,
                                horizontal_flip=True,
                                vertical_flip=True)
        else:
            img = augment_image(img)

        aug_imgs.append(img)
        aug_tabs.append(X_tab[src])
        aug_labels.append(1)

    X_img_out = np.concatenate([X_img, np.array(aug_imgs, dtype=np.float32)], axis=0)
    X_tab_out = np.concatenate([X_tab, np.array(aug_tabs, dtype=np.float32)], axis=0)
    y_out     = np.concatenate([y,     np.array(aug_labels, dtype=np.int32)],  axis=0)

    # Shuffle
    idx = np.random.permutation(len(y_out))
    return X_img_out[idx], X_tab_out[idx], y_out[idx]


# ── HDF5 EXTRACTION ──────────────────────────────────────────────────────────

def extract_images_from_hdf5(hdf5_path: str,
                              output_folder: str,
                              max_images: int | None = None) -> list:
    """
    Extract JPEG images stored inside an ISIC HDF5 file to individual files.

    Parameters
    ----------
    hdf5_path     : path to train-image.hdf5
    output_folder : destination directory
    max_images    : stop after this many images (None = all)

    Returns
    -------
    List of extracted file paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    extracted = []

    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        if max_images:
            keys = keys[:max_images]

        print(f"[extract_images] Extracting {len(keys)} images → {output_folder}")

        for i, key in enumerate(keys):
            try:
                img_data = f[key][()]

                if isinstance(img_data, bytes):
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = img_data

                out_path = os.path.join(output_folder, f"{key}.jpg")
                Image.fromarray(img.astype(np.uint8)).save(out_path, quality=95)
                extracted.append(out_path)

                if (i + 1) % 1000 == 0:
                    print(f"  Extracted {i + 1}/{len(keys)}")

            except Exception as e:
                print(f"[extract_images] SKIP {key}: {e}")

    print(f"[extract_images] Done. {len(extracted)} files saved.")
    return extracted
