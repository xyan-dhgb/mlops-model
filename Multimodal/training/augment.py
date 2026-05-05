"""
augment.py — Hàm augmentation ảnh cho oversampling Malignant
Được import bởi train.py
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance


def augment_image(img_array: np.ndarray,
                  rotation_range: int = 15,
                  brightness_range: tuple = (0.8, 1.2),
                  zoom_range: float = 0.10,
                  h_flip: bool = True,
                  v_flip: bool = False,
                  add_noise: bool = False,
                  noise_sigma: float = 10.0) -> np.ndarray:
    """
    Augmentation ảnh cho oversampling Malignant.
    Strong mode (v_flip=True, add_noise=True) dùng cho tạo mẫu augmented mạnh.
    """
    img = img_array.copy().astype(np.uint8)

    # Random rotation ±rotation_range độ
    angle = np.random.uniform(-rotation_range, rotation_range)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Horizontal flip
    if h_flip and np.random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Vertical flip (strong mode)
    if v_flip and np.random.random() < 0.5:
        img = cv2.flip(img, 0)

    # Brightness scaling
    factor = np.random.uniform(brightness_range[0], brightness_range[1])
    pil    = Image.fromarray(img)
    img    = np.array(ImageEnhance.Brightness(pil).enhance(factor))

    # Zoom crop (0–zoom_range% viền)
    if zoom_range > 0:
        crop_pct = np.random.uniform(0, zoom_range)
        crop_px  = int(min(h, w) * crop_pct)
        if crop_px > 0:
            img = img[crop_px:h - crop_px, crop_px:w - crop_px]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Gaussian noise (strong mode)
    if add_noise:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def oversample_malignant(X_img: np.ndarray,
                          X_tab: np.ndarray,
                          y: np.ndarray,
                          target_ratio: float = 0.25,
                          strong_aug: bool = True):
    """
    Oversample lớp Malignant (y=1) bằng augmentation đến target_ratio.
    target_ratio = 0.25 → 25% mẫu trong tập sau oversampling là Malignant.

    Returns:
        X_img_os, X_tab_os, y_os  (đã shuffle)
    """
    mal_idx = np.where(y == 1)[0]
    ben_idx = np.where(y == 0)[0]

    n_ben = len(ben_idx)
    # Số mẫu Malignant cần sau oversampling
    n_mal_target = int(n_ben * target_ratio / (1 - target_ratio))
    n_to_add     = max(0, n_mal_target - len(mal_idx))

    print(f"Oversampling: {len(mal_idx)} → {len(mal_idx) + n_to_add} Malignant "
          f"(target ratio={target_ratio:.0%})")

    aug_imgs, aug_tabs, aug_ys = [], [], []
    for i in range(n_to_add):
        src_idx = mal_idx[i % len(mal_idx)]
        aug_img = augment_image(
            X_img[src_idx],
            h_flip=True,
            v_flip=strong_aug,
            add_noise=strong_aug,
        )
        aug_imgs.append(aug_img)
        aug_tabs.append(X_tab[src_idx])
        aug_ys.append(1)

    if n_to_add > 0:
        X_img_os = np.concatenate([X_img, np.array(aug_imgs)], axis=0)
        X_tab_os = np.concatenate([X_tab, np.array(aug_tabs)], axis=0)
        y_os     = np.concatenate([y,     np.array(aug_ys)],   axis=0)
    else:
        X_img_os, X_tab_os, y_os = X_img, X_tab, y

    # Shuffle
    perm = np.random.permutation(len(y_os))
    return X_img_os[perm], X_tab_os[perm], y_os[perm]
