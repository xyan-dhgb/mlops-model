import cv2
import numpy as np
def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an X-ray image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Placeholder for missing images
        img = np.zeros(target_size)
    else:
        img = cv2.resize(img, target_size)

    # Normalize the image
    img = img / 255.0

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)

    return img