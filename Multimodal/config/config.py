# Set seeds for reproducibility
import numpy as np
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

# File paths
CSV_PATH = "data/mias_derived_info.csv"
IMAGE_DIR = "data/MIAS"

# Model parameters
TARGET_SIZE = (224, 224)
NUM_CLASSES = 3
EPOCHS = 20
BATCH_SIZE = 32