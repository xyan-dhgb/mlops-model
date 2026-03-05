import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
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

    return y_pred, y_pred_prob