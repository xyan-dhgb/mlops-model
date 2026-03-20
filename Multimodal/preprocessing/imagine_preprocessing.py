import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from data_loader import load_image

def prepare_multimodal_data(df, image_dir, target_size=(224, 224)):
    """
    Prepare the multimodal dataset combining images and CSV data
    """
    # Extract relevant features from CSV
    X_tabular = df[['BG', 'CLASS_NUM', 'X', 'Y', 'RADIUS', 'DENSITY_NUM', 'BIRADS_NUM']].values

    # Normalize tabular data
    scaler = StandardScaler()
    X_tabular = scaler.fit_transform(X_tabular)

    # Target variable
    y = df['SEVERITY_NUM'].values

    # Convert to categorical
    y_cat = to_categorical(y)

    # Prepare image data
    X_images = []
    for ref_num in df['REFNUM'].values:
        image_path = os.path.join(image_dir, f"{ref_num}.pgm")
        img = load_image(image_path, target_size)
        X_images.append(img)

    X_images = np.array(X_images)

    # Split the dataset
    X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
        X_tabular, X_images, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    return X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test