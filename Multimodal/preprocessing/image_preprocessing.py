"""
Image Preprocessing for ISIC 2024
Covers: load, CLAHE, Gaussian blur, contrast enhancement, augmentation.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path


# ── Core helpers ─────────────────────────────────────────────────────────────

def load_image(
    image_path: str,
    target_size: tuple = (224, 224),
    color_mode: str = "rgb"
) -> np.ndarray | None:
    """
    Load and resize an image file.

    Args:
        image_path:  Path to the image.
        target_size: (width, height) to resize to.
        color_mode:  'rgb', 'grayscale', or 'rgba'.

    Returns:
        float32 numpy array in [0, 1], or None on error.
    """
    try:
        img = Image.open(image_path)
        mode_map = {"rgb": "RGB", "grayscale": "L", "rgba": "RGBA"}
        img = img.convert(mode_map.get(color_mode.lower(), "RGB"))
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception as exc:
        print(f"[load_image] Error loading {image_path}: {exc}")
        return None


def preprocess_image(
    img_array: np.ndarray,
    apply_clahe: bool = True,
    apply_gaussian: bool = True,
    enhance_contrast: float = 1.2
) -> np.ndarray | None:
    """
    Apply CLAHE, Gaussian blur, and contrast enhancement.

    Args:
        img_array:        float32 array in [0, 1], shape (H, W, 3).
        apply_clahe:      Apply CLAHE on LAB L-channel.
        apply_gaussian:   Apply mild Gaussian blur (noise reduction).
        enhance_contrast: Pillow contrast factor (1.0 = no change).

    Returns:
        Processed float32 array in [0, 1].
    """
    if img_array is None:
        return None
    try:
        img_u8 = (img_array * 255).astype(np.uint8)

        if apply_clahe and img_u8.ndim == 3:
            lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if apply_gaussian:
            img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)

        if enhance_contrast != 1.0:
            pil_img = Image.fromarray(img_u8)
            pil_img = ImageEnhance.Contrast(pil_img).enhance(enhance_contrast)
            img_u8 = np.array(pil_img)

        return img_u8.astype(np.float32) / 255.0
    except Exception as exc:
        print(f"[preprocess_image] Error: {exc}")
        return None


def augment_image(
    img_array: np.ndarray,
    rotation_range: int = 15,
    brightness_range: tuple = (0.8, 1.2),
    zoom_range: float = 0.1,
    horizontal_flip: bool = True
) -> np.ndarray:
    """
    Apply random augmentation: rotation, brightness, zoom, flip.

    Args:
        img_array:        float32 array in [0, 1].
        rotation_range:   Max rotation in degrees.
        brightness_range: (min, max) brightness factors.
        zoom_range:       Max zoom fraction.
        horizontal_flip:  Whether to randomly flip horizontally.

    Returns:
        Augmented float32 array in [0, 1].
    """
    img_u8 = (img_array * 255).astype(np.uint8)
    h, w = img_u8.shape[:2]

    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random brightness
    factor = np.random.uniform(*brightness_range)
    pil_img = ImageEnhance.Brightness(Image.fromarray(img_u8)).enhance(factor)
    img_u8 = np.array(pil_img)

    # Random zoom (center crop + resize back)
    if zoom_range > 0:
        zf = np.random.uniform(0, zoom_range)
        crop_h = int(h * (1 - zf))
        crop_w = int(w * (1 - zf))
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        img_u8 = img_u8[start_y:start_y + crop_h, start_x:start_x + crop_w]
        img_u8 = cv2.resize(img_u8, (w, h))

    # Random horizontal flip
    if horizontal_flip and np.random.random() > 0.5:
        img_u8 = cv2.flip(img_u8, 1)

    return img_u8.astype(np.float32) / 255.0


# ── HDF5 extraction (ISIC 2024 stores images in HDF5) ────────────────────────

def extract_images_from_hdf5(
    hdf5_path: str,
    output_folder: str,
    max_images: int | None = None
) -> str:
    """
    Extract JPEG images from ISIC 2024's train-image.hdf5 to a directory.

    Args:
        hdf5_path:     Path to the HDF5 file.
        output_folder: Directory to write extracted .jpg files.
        max_images:    Limit extraction (None = all).

    Returns:
        output_folder path.
    """
    import h5py
    from pathlib import Path

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Extracting images from: {hdf5_path}")

    extracted = errors = 0
    with h5py.File(hdf5_path, "r") as hf:
        keys = list(hf.keys())
        if max_images:
            keys = keys[:max_images]
        print(f"Total images in HDF5: {len(hf.keys())} | Extracting: {len(keys)}")

        for i, isic_id in enumerate(keys):
            try:
                img_bytes = hf[isic_id][()]
                out_path = Path(output_folder) / f"{isic_id}.jpg"
                out_path.write_bytes(img_bytes)
                extracted += 1
                if (i + 1) % 1000 == 0:
                    print(f"  Extracted {i + 1}/{len(keys)}")
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"  Error with {isic_id}: {exc}")

    print(f"Done. Extracted: {extracted}, Errors: {errors}")
    return output_folder
