"""
Configuration module for Human Activity Recognition (HAR) CNN-LSTM Project.
Defines hyperparameter defaults, file paths, labels, and device configuration.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "UCI HAR Dataset")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")

# Create directories if they do not exist
for dir_path in [DATA_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR, SAMPLE_DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset download URLs (Primary and Fallback mirror)
UCI_HAR_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
]
DATASET_ZIP_PATH = os.path.join(DATA_DIR, "UCI_HAR_Dataset.zip")

# Activity Classes mapping
ACTIVITY_LABEL_MAP = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

ACTIVITY_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

# Modern cyberpunk / neon UI color palette for classes
ACTIVITY_COLORS = {
    "Walking": "#38bdf8",          # Electric Cyan
    "Walking Upstairs": "#6366f1", # Indigo Neon
    "Walking Downstairs": "#818cf8",# Periwinkle
    "Sitting": "#ec4899",          # Neon Pink
    "Standing": "#a855f7",         # Electric Violet
    "Laying": "#10b981"            # Emerald Neon
}

# Raw Inertial Signal Filenames (9 channels recorded at 50Hz, 128 samples per window)
INERTIAL_SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

# Model and Data Hyperparameters
SEQ_LEN = 128          # 128 time-series steps (2.56s at 50 Hz)
NUM_CHANNELS = 9       # 9 sensor channels
NUM_CLASSES = 6        # 6 activity categories

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10          # Early stopping patience
VAL_SPLIT = 0.15       # Split fraction from train set for validation

# Default saved model and evaluation metrics paths
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_cnn_lstm.pt")
FINAL_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "final_cnn_lstm.pt")
METRICS_JSON_PATH = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
TRAIN_HISTORY_PATH = os.path.join(OUTPUTS_DIR, "training_history.json")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
ROC_CURVES_PATH = os.path.join(OUTPUTS_DIR, "loss_accuracy_curves.png")

# Computation Device configuration
def get_device():
    """Detect and return PyTorch compute device (CUDA GPU or CPU fallback)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

