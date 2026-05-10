"""
augment.py — Augmentation + Oversampling Malignant
Khớp notebook cell 28 (augment_image) + cell 36 (oversample_malignant).
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance


def augment_image(img_array: np.ndarray,
                  rotation_range: int = 15,
                  brightness_range: tuple = (0.8, 1.2),
                  zoom_range: float = 0.10,
                  strong: bool = False) -> np.ndarray:
    """
    Khớp notebook cell 28 augment_image():
      strong=True  → rotation ±30°, flip ngang+dọc, noise (Malignant)
      strong=False → rotation ±15°, flip ngang only
    """
    img = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.copy()
    h, w = img.shape[:2]

    # Rotation
    angle = np.random.uniform(-30, 30) if strong else np.random.uniform(-rotation_range, rotation_range)
    M   = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Flip ngang
    if np.random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Flip dọc (strong mode)
    if strong and np.random.random() < 0.5:
        img = cv2.flip(img, 0)

    # Brightness
    factor = np.random.uniform(brightness_range[0], brightness_range[1])
    img = np.array(ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor))

    # Zoom crop
    if zoom_range > 0:
        crop = int(min(h, w) * np.random.uniform(0, zoom_range))
        if crop > 0:
            img = img[crop:h - crop, crop:w - crop]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    return img.astype(np.float32) / 255.0


def oversample_malignant(X_img: np.ndarray,
                          X_tab: np.ndarray,
                          y: np.ndarray,
                          target_ratio: float = 0.25,
                          strong_aug: bool = True):
    """
    Oversample Malignant bằng augmentation đến target_ratio.
    Khớp notebook cell 36: target_ratio=0.25, strong_aug=True.
    """
    mal_idx = np.where(y == 1)[0]
    n_ben   = int((y == 0).sum())
    n_mal_target = int(n_ben * target_ratio / (1 - target_ratio))
    n_to_add     = max(0, n_mal_target - len(mal_idx))

    print(f"Oversampling Malignant: {len(mal_idx)} → {len(mal_idx) + n_to_add} "
          f"(target_ratio={target_ratio:.0%})")

    aug_imgs, aug_tabs, aug_ys = [], [], []
    for i in range(n_to_add):
        src = mal_idx[i % len(mal_idx)]
        aug_imgs.append(augment_image(X_img[src], strong=strong_aug))
        aug_tabs.append(X_tab[src])
        aug_ys.append(1)

    if n_to_add > 0:
        X_img = np.concatenate([X_img, np.array(aug_imgs)], axis=0)
        X_tab = np.concatenate([X_tab, np.array(aug_tabs)], axis=0)
        y     = np.concatenate([y,     np.array(aug_ys)],   axis=0)

    perm = np.random.permutation(len(y))
    return X_img[perm], X_tab[perm], y[perm]
