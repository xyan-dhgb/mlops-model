import cv2
import numpy as np
import matplotlib.pyplot as plt
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