"""
Data loading and preprocessing module for UCI Human Activity Recognition.
Loads raw 9-channel inertial sensor sequences, applies train-fitted channel standardization,
and constructs PyTorch DataLoaders without data leakage.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import (
    RAW_DATA_DIR,
    SAVED_MODELS_DIR,
    SAMPLE_DATA_DIR,
    INERTIAL_SIGNAL_NAMES,
    ACTIVITY_LABEL_MAP,
    ACTIVITY_NAMES,
    BATCH_SIZE,
    RANDOM_SEED,
    VAL_SPLIT
)
from data.download_dataset import ensure_dataset


class HARDataset(Dataset):
    """PyTorch Dataset for Human Activity Recognition inertial time-series."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray = None):
        """
        Args:
            signals: Array of shape (N, 9, 128) representing 9 inertial sensor channels.
            labels: Array of shape (N,) with integer class labels (0 to 5).
        """
        self.signals = torch.tensor(signals, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.signals[idx], self.labels[idx]
        return self.signals[idx]


def load_raw_inertial_signals(subset: str = "train", raw_data_dir: str = RAW_DATA_DIR) -> np.ndarray:
    """
    Load all 9 raw inertial signal text files for the given subset ('train' or 'test').
    Returns:
        np.ndarray: shape (N_samples, 9, 128)
    """
    subset_dir = os.path.join(raw_data_dir, subset, "Inertial Signals")
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"Inertial Signals folder not found at: {subset_dir}")

    channels_data = []
    for signal_name in INERTIAL_SIGNAL_NAMES:
        filename = f"{signal_name}_{subset}.txt"
        file_path = os.path.join(subset_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing signal file: {file_path}")

        # Fast loading using pandas whitespace-delimited parser
        df = pd.read_csv(file_path, sep=r"\s+", header=None, dtype=np.float32)
        # df shape: (N_samples, 128)
        channels_data.append(df.values)

    # Stack along channel axis -> shape (9, N_samples, 128)
    stacked = np.stack(channels_data, axis=0)
    # Permute to (N_samples, 9, 128)
    signals = np.transpose(stacked, (1, 0, 2))
    return signals


def load_labels(subset: str = "train", raw_data_dir: str = RAW_DATA_DIR) -> np.ndarray:
    """
    Load ground-truth labels for subset ('train' or 'test').
    Labels are 1-based in UCI HAR (1 to 6); converted to 0-based (0 to 5).
    """
    label_path = os.path.join(raw_data_dir, subset, f"y_{subset}.txt")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    y = pd.read_csv(label_path, sep=r"\s+", header=None).values.squeeze()
    # Convert from 1..6 to 0..5
    y_zero_indexed = (y - 1).astype(np.int64)
    return y_zero_indexed


def load_subjects(subset: str = "train", raw_data_dir: str = RAW_DATA_DIR) -> np.ndarray:
    """Load subject IDs for subset ('train' or 'test')."""
    subj_path = os.path.join(raw_data_dir, subset, f"subject_{subset}.txt")
    if not os.path.exists(subj_path):
        return None
    return pd.read_csv(subj_path, sep=r"\s+", header=None).values.squeeze()


def load_engineered_features(subset: str = "train", raw_data_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load the 561 hand-crafted statistical features (X_train.txt or X_test.txt).
    """
    features_txt = os.path.join(raw_data_dir, "features.txt")
    feature_names = []
    if os.path.exists(features_txt):
        with open(features_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    feature_names.append(parts[1])

    x_path = os.path.join(raw_data_dir, subset, f"X_{subset}.txt")
    df = pd.read_csv(x_path, sep=r"\s+", header=None, dtype=np.float32)
    if len(feature_names) == df.shape[1]:
        df.columns = feature_names
    return df


class SensorScaler:
    """
    Standardizes 9-channel sensor sequences across time steps.
    Computes per-channel mean and std strictly on training set to avoid data leakage.
    """

    def __init__(self):
        self.means = None  # shape (1, 9, 1)
        self.stds = None   # shape (1, 9, 1)

    def fit(self, X: np.ndarray):
        """Fit scaler on training data of shape (N, 9, 128)."""
        self.means = np.mean(X, axis=(0, 2), keepdims=True)
        self.stds = np.std(X, axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize data using fitted parameters."""
        if self.means is None or self.stds is None:
            raise ValueError("SensorScaler must be fitted before transforming data.")
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: str):
        """Save scaler parameters to JSON."""
        data = {
            "means": self.means.squeeze().tolist(),
            "stds": self.stds.squeeze().tolist()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, filepath: str):
        """Load scaler parameters from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.means = np.array(data["means"], dtype=np.float32).reshape(1, -1, 1)
        self.stds = np.array(data["stds"], dtype=np.float32).reshape(1, -1, 1)
        return self


def get_data_loaders(
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    random_seed: int = RANDOM_SEED,
    raw_data_dir: str = RAW_DATA_DIR
):
    """
    Load raw inertial data, perform stratified train-validation split,
    normalize without leakage, and return PyTorch DataLoaders.
    """
    ensure_dataset(raw_data_dir)

    print("Loading raw inertial time-series data...")
    X_train_raw = load_raw_inertial_signals("train", raw_data_dir)  # (7352, 9, 128)
    y_train_raw = load_labels("train", raw_data_dir)                # (7352,)

    X_test_raw = load_raw_inertial_signals("test", raw_data_dir)    # (2947, 9, 128)
    y_test = load_labels("test", raw_data_dir)                      # (2947,)

    print(f"Raw Train samples: {X_train_raw.shape[0]} | Test samples: {X_test_raw.shape[0]}")

    # Stratified Train / Validation Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=random_seed)
    train_idx, val_idx = next(sss.split(X_train_raw, y_train_raw))

    X_train = X_train_raw[train_idx]
    y_train = y_train_raw[train_idx]

    X_val = X_train_raw[val_idx]
    y_val = y_train_raw[val_idx]

    print(f"Split sizes -> Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test_raw.shape[0]}")

    # Fit Scaler strictly on train split
    scaler = SensorScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test_raw)

    # Save scaler parameters for inference
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler_params.json")
    scaler.save(scaler_path)
    print(f"Saved fitted SensorScaler parameters to: {scaler_path}")

    # Create PyTorch datasets
    train_dataset = HARDataset(X_train_norm, y_train)
    val_dataset = HARDataset(X_val_norm, y_val)
    test_dataset = HARDataset(X_test_norm, y_test)

    # PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, scaler


def generate_sample_recordings(raw_data_dir: str = RAW_DATA_DIR, output_dir: str = SAMPLE_DATA_DIR):
    """
    Extract single 128-step sensor windows from the test set for each of the 6 activities.
    Saves clean CSV files that users can upload or preview in the Streamlit UI.
    Each CSV is formatted as 128 rows x 9 columns with appropriate header names.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_test = load_raw_inertial_signals("test", raw_data_dir)
    y_test = load_labels("test", raw_data_dir)

    for class_id, class_name in ACTIVITY_LABEL_MAP.items():
        # Find matches for this activity
        indices = np.where(y_test == class_id)[0]
        if len(indices) > 0:
            sample_idx = indices[0]
            # shape: (9, 128) -> transpose to (128, 9)
            sample_window = X_test[sample_idx].T  # (128, 9)
            df = pd.DataFrame(sample_window, columns=INERTIAL_SIGNAL_NAMES)
            
            clean_name = class_name.lower().replace(" ", "_")
            csv_path = os.path.join(output_dir, f"sample_{clean_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Generated sample CSV: {csv_path} (Shape: {df.shape})")

    # Also generate a multi-window continuous sequence CSV for demonstration
    multi_window_indices = np.random.RandomState(42).choice(len(X_test), 5, replace=False)
    combined_windows = []
    for idx in multi_window_indices:
        combined_windows.append(X_test[idx].T)
    multi_df = pd.DataFrame(np.vstack(combined_windows), columns=INERTIAL_SIGNAL_NAMES)
    multi_path = os.path.join(output_dir, "sample_multi_window_recording.csv")
    multi_df.to_csv(multi_path, index=False)
    print(f"Generated multi-window recording: {multi_path} (Shape: {multi_df.shape})")


if __name__ == "__main__":
    generate_sample_recordings()

