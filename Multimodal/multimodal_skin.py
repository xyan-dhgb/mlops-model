import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from eli5.sklearn import PermutationImportance
import eli5

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_csv_data(file_path):
    """
    Load the CSV data and display basic information
    """
    print("Loading CSV data...")
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    """
    Explore and visualize the dataset
    """
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    # Check unique values in each column
    print("\nUnique values in each column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Visualize class and severity distributions
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    class_counts = df['CLASS'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Classes')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    severity_counts = df['SEVERITY'].value_counts()
    sns.barplot(x=severity_counts.index, y=severity_counts.values)
    plt.title('Distribution of Severity')

    plt.tight_layout()
    plt.show()

    # Visualize breast density distribution
    plt.figure(figsize=(8, 5))
    density_counts = df['DENSITY'].value_counts()
    sns.barplot(x=density_counts.index, y=density_counts.values)
    plt.title('Distribution of Breast Density')
    plt.show()

    # Visualize relationship between class and severity
    plt.figure(figsize=(12, 6))
    sns.countplot(x='CLASS', hue='SEVERITY', data=df)
    plt.title('Class vs Severity')
    plt.xticks(rotation=45)
    plt.show()

def preprocess_csv_data(df):
    """
    Preprocess the CSV data for model input
    """
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()

    # Fill missing coordinates with -1 (for normal cases)
    for col in ['X', 'Y', 'RADIUS']:
        processed_df[col] = processed_df[col].fillna(-1)

    # Convert categorical variables to numeric
    # Background tissue
    bg_map = {'F': 0, 'G': 1, 'D': 2}
    processed_df['BG'] = processed_df['BG'].map(bg_map)

    # Class mapping
    class_map = {
        'NORM': 0,
        'CIRC': 1,
        'SPIC': 2,
        'ARCH': 3,
        'ASYM': 4,
        'CALC': 5,
        'MISC': 6
    }
    processed_df['CLASS_NUM'] = processed_df['CLASS'].map(class_map)

    # Severity mapping
    severity_map = {'Normal': 0, 'Benign': 1, 'Malignant': 2}
    processed_df['SEVERITY_NUM'] = processed_df['SEVERITY'].map(severity_map)

    # Density mapping
    density_map = {'A': 1, 'B': 2, 'C/D': 3}
    processed_df['DENSITY_NUM'] = processed_df['DENSITY'].map(density_map)

    # BI-RADS mapping
    birads_map = {
        'BI-RADS 1': 1,
        'BI-RADS 2': 2,
        'BI-RADS 3': 3,
        'BI-RADS 4': 4,
        'BI-RADS 5': 5
    }
    processed_df['BIRADS_NUM'] = processed_df['BI-RADS'].map(birads_map)

    return processed_df

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

def build_multimodal_model(tabular_shape, image_shape, num_classes=3):
    """
    Build a multimodal model that combines tabular data and images
    """
    # Tabular data branch
    tabular_input = Input(shape=(tabular_shape,), name='tabular_input')
    x_tab = Dense(64, activation='relu')(tabular_input)
    x_tab = Dropout(0.3)(x_tab)
    x_tab = Dense(32, activation='relu')(x_tab)
    x_tab = Model(inputs=tabular_input, outputs=x_tab)

    # Image branch using pretrained model
    image_input = Input(shape=image_shape, name='image_input')

    # Convert greyscale to RGB by repeating the channel
    x_img = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(image_input)

    # Use pretrained ResNet with weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model
    base_model.trainable = False

    x_img = base_model(x_img)
    x_img = Flatten()(x_img)
    x_img = Dense(128, activation='relu')(x_img)
    x_img = Dropout(0.5)(x_img)
    x_img = Model(inputs=image_input, outputs=x_img)

    # Combine the branches
    combined = concatenate([x_tab.output, x_img.output])

    # Final layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=[x_tab.input, x_img.input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, X_tab_train, X_img_train, y_train, X_tab_val, X_img_val, y_val, epochs=20, batch_size=32):
    """
    Train the multimodal model
    """
    # Data augmentation for images
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        [X_tab_train, X_img_train],
        y_train,
        validation_data=([X_tab_val, X_img_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    return model, history

def evaluate_model(model, X_tab_test, X_img_test, y_test):
    """
    Evaluate the model's performance
    """
    # Get predictions
    y_pred_prob = model.predict([X_tab_test, X_img_test])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Benign', 'Malignant']))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Benign', 'Malignant'],
                yticklabels=['Normal', 'Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    class_names = ['Normal', 'Benign', 'Malignant']

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

    return y_pred, y_pred_prob

def predict_cancer_diagnosis(model, csv_data, image_path):
    """
    Make predictions using both CSV data and X-ray image
    """
    # Process CSV data
    processed_csv = preprocess_csv_data(pd.DataFrame([csv_data]))

    # Add 'SEVERITY_NUM' column if it doesn't exist (bugfix) and set to 0 (Normal)
    if 'SEVERITY_NUM' not in processed_csv.columns:
        processed_csv['SEVERITY_NUM'] = 0

    X_tab = processed_csv[['BG', 'CLASS_NUM', 'X', 'Y', 'RADIUS', 'DENSITY_NUM', 'BIRADS_NUM']].values

    # Load and process the image
    X_img = np.expand_dims(load_image(image_path), axis=0)

    # Make prediction
    prediction = model.predict([X_tab, X_img])
    pred_class = np.argmax(prediction, axis=1)[0]

    # Map back to diagnosis
    diagnosis_map = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
    diagnosis = diagnosis_map[pred_class]

    # Calculate confidence
    confidence = prediction[0][pred_class] * 100

    return {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'probabilities': {
            'Normal': float(prediction[0][0] * 100),
            'Benign': float(prediction[0][1] * 100),
            'Malignant': float(prediction[0][2] * 100)
        }
    }

class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class to make the model compatible with scikit-learn APIs
    """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self  

    def predict(self, X):
        # Create dummy image data of appropriate shape
        dummy_img = np.zeros((X.shape[0], 224, 224, 1))
        return np.argmax(self.model.predict([X, dummy_img]), axis=1)

    def score(self, X, y):
        # Convert y to multiclass format if necessary
        if y.ndim == 2 and y.shape[1] > 1:  # Check if y is one-hot encoded
            y = np.argmax(y, axis=1)

        # Predict using the wrapper
        y_pred = self.predict(X)

        # Calculate accuracy
        return accuracy_score(y, y_pred)

def analyze_feature_importance(model, X_tab_test, y_test):
    """
    Analyze the importance of tabular features
    """
    # Wrap the original model
    wrapped_model = ModelWrapper(model)

    # Calculate permutation importance using the wrapped model
    perm = PermutationImportance(wrapped_model, random_state=42).fit(
        X_tab_test, y_test
    )

    # Display feature importance
    feature_names = ['BG', 'CLASS', 'X', 'Y', 'RADIUS', 'DENSITY', 'BI-RADS']
    print("Feature importance analysis:")
    display = eli5.show_weights(perm, feature_names=feature_names)
    return display

def visualize_diagnosis(image_path, prediction_result, heatmap=None):
    """
    Create a visualization that highlights suspicious areas in the mammogram
    along with the diagnosis results
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((512, 512))
    else:
        img = cv2.resize(img, (512, 512))

    plt.figure(figsize=(14, 7))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Mammogram')
    plt.axis('off')

    # Display image with highlighted regions (if heatmap)
    plt.subplot(1, 2, 2)
    if heatmap is not None:
        # Apply heatmap
        heatmap = cv2.resize(heatmap, (512, 512))
        plt.imshow(img, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.title('Suspicious Areas')
    else:
        plt.imshow(img, cmap='gray')
        plt.title('No Heatmap Available')
    plt.axis('off')

    # Add diagnosis information
    diagnosis = prediction_result['diagnosis']
    confidence = prediction_result['confidence']
    plt.figtext(0.5, 0.01, f"Diagnosis: {diagnosis} (Confidence: {confidence:.2f}%)",
                ha="center", fontsize=14,
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def demo_visualize_diagnosis():
    """
    Demonstrate the visualization function with sample data
    """
    # Sample image path - replace with your actual path
    image_path = "sample_mammogram.png"  

    # Create a sample prediction result dictionary
    prediction_result = {
        'diagnosis': 'Benign',
        'confidence': 85.7,
        'probabilities': {
            'Normal': 10.2,
            'Benign': 85.7,
            'Malignant': 4.1
        }
    }

    # Call the function (without heatmap)
    visualize_diagnosis(image_path, prediction_result)

    # Create a simple heatmap for demonstration:
    heatmap = np.zeros((512, 512))
    # Create a "hotspot" in the heatmap
    heatmap[200:300, 200:300] = np.linspace(0, 1, 100)[:, np.newaxis] * np.linspace(0, 1, 100)[np.newaxis, :]

    # Then call with heatmap
    visualize_diagnosis(image_path, prediction_result, heatmap)


## file main.py 
def main():
    """
    Main function to execute the complete workflow
    """
    print("=== Breast Cancer Detection System with Multimodal Learning ===")
    
    # Set file paths - replace with your actual paths
    csv_path = "data/mias_derived_info.csv"
    image_dir = "data/MIAS"
    
    # 1. Load and explore data
    try:
        df = load_csv_data(csv_path)
        explore_data(df)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        print("Using a small sample dataframe for demonstration purposes.")
        # Create a small sample dataframe for demonstration
        sample_data = {
            'REFNUM': ['mdb001', 'mdb002', 'mdb003'],
            'BG': ['G', 'F', 'G'],
            'CLASS': ['NORM', 'CIRC', 'SPIC'],
            'X': [None, 540, 470],
            'Y': [None, 565, 480],
            'RADIUS': [None, 60, 45],
            'DENSITY': ['B', 'A', 'B'],
            'BI-RADS': ['BI-RADS 1', 'BI-RADS 3', 'BI-RADS 4'],
            'SEVERITY': ['Normal', 'Benign', 'Malignant']
        }
        df = pd.DataFrame(sample_data)
        print("Created sample dataframe:")
        print(df)
    
    # 2. Preprocess data
    processed_df = preprocess_csv_data(df)
    print("\nPreprocessed dataframe:")
    print(processed_df.head())
    
    # 3. Prepare multimodal data
    print("\nPreparing multimodal data...")
    try:
        X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = prepare_multimodal_data(
            processed_df, image_dir
        )
        print(f"Training data shapes: Tabular: {X_tab_train.shape}, Images: {X_img_train.shape}")
    except:
        print("Warning: Could not prepare actual image data. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        X_tab_train = np.random.randn(80, 7)
        X_tab_test = np.random.randn(20, 7)
        X_img_train = np.zeros((80, 224, 224, 1))
        X_img_test = np.zeros((20, 224, 224, 1))
        y_train = to_categorical(np.random.randint(0, 3, 80))
        y_test = to_categorical(np.random.randint(0, 3, 20))
    
    # 4. Build model
    print("\nBuilding multimodal model...")
    model = build_multimodal_model(
        tabular_shape=X_tab_train.shape[1],
        image_shape=X_img_train.shape[1:],
        num_classes=3
    )
    model.summary()
    
    # 5. Train model
    print("\nTraining model...")
    trained_model, history = train_model(
        model,
        X_tab_train, X_img_train, y_train,
        X_tab_test, X_img_test, y_test,
        epochs=5  # Reduced for demonstration
    )
    
    # 6. Evaluate model
    print("\nEvaluating model...")
    y_pred, y_pred_prob = evaluate_model(trained_model, X_tab_test, X_img_test, y_test)
    
    # 7. Feature importance analysis
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(trained_model, X_tab_test, y_test)
    
    # 8. Sample prediction
    print("\nMaking a sample prediction...")
    new_patient = {
        'REFNUM': 'new_patient_001',
        'BG': 'G',
        'CLASS': 'CIRC',
        'X': 520,
        'Y': 380,
        'RADIUS': 45,
        'DENSITY': 'B',
        'BI-RADS': 'BI-RADS 3',
        'SEVERITY': 'Benign'  # May need to adjust this value based on expected input
    }
    
    # Sample image path for the new patient
    sample_image_path = os.path.join(image_dir, "sample.pgm")
    if not os.path.exists(sample_image_path):
        print(f"Warning: Image file '{sample_image_path}' not found. Using zero array.")
        # Create a dummy image file for demonstration
        dummy_img = np.zeros((224, 224))
        cv2.imwrite("dummy_sample.pgm", dummy_img)
        sample_image_path = "dummy_sample.pgm"
        
    # Make prediction
    diagnosis_result = predict_cancer_diagnosis(trained_model, new_patient, sample_image_path)
    print(f"Diagnosis: {diagnosis_result['diagnosis']}")
    print(f"Confidence: {diagnosis_result['confidence']:.2f}%")
    print("Probabilities:")
    for class_name, prob in diagnosis_result['probabilities'].items():
        print(f"  {class_name}: {prob:.2f}%")
    
    # 9. Visualize the result
    print("\nVisualizing diagnosis result...")
    try:
        visualize_diagnosis(sample_image_path, diagnosis_result)
        
        # Create a sample heatmap
        heatmap = np.zeros((512, 512))
        heatmap[200:300, 200:300] = np.linspace(0, 1, 100)[:, np.newaxis] * np.linspace(0, 1, 100)[np.newaxis, :]
        visualize_diagnosis(sample_image_path, diagnosis_result, heatmap)
    except Exception as e:
        print(f"Error visualizing diagnosis: {e}")
    
    print("\nBreast Cancer Detection workflow completed successfully!")

if __name__ == "__main__":
    main()