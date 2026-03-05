# main.py
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from config import CSV_PATH, IMAGE_DIR, TARGET_SIZE, NUM_CLASSES
from data_loader import load_csv_data, load_image
from preprocessing import preprocess_csv_data, prepare_multimodal_data
from models import build_multimodal_model
from training import train_model, evaluate_model
from utils import visualize_diagnosis


def predict_cancer_diagnosis(model, patient_data, image_path):
    bg_map = {'F': 0, 'G': 1, 'D': 2}
    class_map = {'NORM': 0, 'CIRC': 1, 'SPIC': 2, 'ARCH': 3, 'ASYM': 4, 'CALC': 5, 'MISC': 6}
    density_map = {'A': 1, 'B': 2, 'C/D': 3}
    birads_map = {'BI-RADS 1': 1, 'BI-RADS 2': 2, 'BI-RADS 3': 3, 'BI-RADS 4': 4, 'BI-RADS 5': 5}

    x_tab = np.array([[bg_map[patient_data['BG']], class_map[patient_data['CLASS']], patient_data['X'],
                       patient_data['Y'], patient_data['RADIUS'],
                       density_map[patient_data['DENSITY']], birads_map[patient_data['BI-RADS']]]])
    x_tab = StandardScaler().fit_transform(x_tab)
    x_img = load_image(image_path, target_size=TARGET_SIZE)
    x_img = np.expand_dims(x_img, axis=0)
    prediction = model.predict([x_tab, x_img])[0]
    class_names = ['Normal', 'Benign', 'Malignant']
    pred_class = np.argmax(prediction)
    result = {
        'diagnosis': class_names[pred_class],
        'confidence': prediction[pred_class] * 100,
        'probabilities': {name: prob * 100 for name, prob in zip(class_names, prediction)}
    }
    return result


def main():
    print("=== Breast Cancer Detection System with Multimodal Learning ===")

    try:
        df = load_csv_data(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_PATH}' not found.")
        print("Using a small sample dataframe for demonstration purposes.")
        df = pd.DataFrame({
            'REFNUM': ['mdb001', 'mdb002', 'mdb003'],
            'BG': ['G', 'F', 'G'],
            'CLASS': ['NORM', 'CIRC', 'SPIC'],
            'X': [None, 540, 470],
            'Y': [None, 565, 480],
            'RADIUS': [None, 60, 45],
            'DENSITY': ['B', 'A', 'B'],
            'BI-RADS': ['BI-RADS 1', 'BI-RADS 3', 'BI-RADS 4'],
            'SEVERITY': ['Normal', 'Benign', 'Malignant']
        })

    processed_df = preprocess_csv_data(df)
    print("\nPreprocessed dataframe:")
    print(processed_df.head())

    print("\nPreparing multimodal data...")
    try:
        X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = prepare_multimodal_data(
            processed_df, IMAGE_DIR, TARGET_SIZE
        )
    except:
        print("Warning: Could not prepare actual image data. Using dummy data.")
        X_tab_train = np.random.randn(80, 7)
        X_tab_test = np.random.randn(20, 7)
        X_img_train = np.zeros((80, *TARGET_SIZE, 1))
        X_img_test = np.zeros((20, *TARGET_SIZE, 1))
        y_train = to_categorical(np.random.randint(0, NUM_CLASSES, 80))
        y_test = to_categorical(np.random.randint(0, NUM_CLASSES, 20))

    print("\nBuilding multimodal model...")
    model = build_multimodal_model(X_tab_train.shape[1], X_img_train.shape[1:], NUM_CLASSES)
    model.summary()

    print("\nTraining model...")
    model, history = train_model(model, X_tab_train, X_img_train, y_train, X_tab_test, X_img_test, y_test,
                                 epochs=5, batch_size=32)

    print("\nEvaluating model...")
    y_pred, y_pred_prob = evaluate_model(model, X_tab_test, X_img_test, y_test)

    print("\nSample prediction...")
    new_patient = {
        'REFNUM': 'new_patient_001',
        'BG': 'G',
        'CLASS': 'CIRC',
        'X': 520,
        'Y': 380,
        'RADIUS': 45,
        'DENSITY': 'B',
        'BI-RADS': 'BI-RADS 3',
        'SEVERITY': 'Benign'
    }

    sample_image_path = os.path.join(IMAGE_DIR, "sample.pgm")
    if not os.path.exists(sample_image_path):
        dummy_img = np.zeros(TARGET_SIZE)
        cv2.imwrite("dummy_sample.pgm", dummy_img)
        sample_image_path = "dummy_sample.pgm"

    result = predict_cancer_diagnosis(model, new_patient, sample_image_path)
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.2f}%")

    print("\nVisualizing diagnosis result...")
    try:
        visualize_diagnosis(sample_image_path, result)
        heatmap = np.zeros((512, 512))
        heatmap[200:300, 200:300] = np.outer(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        visualize_diagnosis(sample_image_path, result, heatmap)
    except Exception as e:
        print(f"Visualization error: {e}")

    print("\nWorkflow complete!")


if __name__ == "__main__":
    main()
