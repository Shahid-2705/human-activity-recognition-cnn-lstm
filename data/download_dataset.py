"""
Dataset download and extraction module for UCI Human Activity Recognition.
Handles automatic downloading, mirror fallback, nested zip extraction, and directory validation.
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    DATASET_ZIP_PATH,
    UCI_HAR_ZIP_URLS
)


def is_dataset_ready(raw_data_dir=RAW_DATA_DIR) -> bool:
    """
    Check if the UCI HAR dataset is already extracted and all required
    directories/files exist.
    """
    if not os.path.isdir(raw_data_dir):
        return False

    required_items = [
        "activity_labels.txt",
        "features.txt",
        os.path.join("train", "X_train.txt"),
        os.path.join("train", "y_train.txt"),
        os.path.join("train", "Inertial Signals"),
        os.path.join("test", "X_test.txt"),
        os.path.join("test", "y_test.txt"),
        os.path.join("test", "Inertial Signals")
    ]

    for item in required_items:
        full_path = os.path.join(raw_data_dir, item)
        if not os.path.exists(full_path):
            return False

    return True


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with a progress bar and streaming chunks."""
    try:
        print(f"Connecting to: {url}")
        response = requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            print(f"Failed to fetch {url}: HTTP status {response.status_code}")
            return False

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 64  # 64 KB

        with open(dest_path, "wb") as f, tqdm(
            desc="Downloading dataset",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"Downloaded successfully: {dest_path}")
        return True
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Safely extract zip archive and handle potential nested zips."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # Some UCI distributions package an inner "UCI HAR Dataset.zip"
    inner_zip = os.path.join(extract_to, "UCI HAR Dataset.zip")
    if os.path.exists(inner_zip):
        print(f"Found nested archive {inner_zip}. Extracting...")
        with zipfile.ZipFile(inner_zip, "r") as inner_zf:
            inner_zf.extractall(extract_to)
        try:
            os.remove(inner_zip)
        except OSError:
            pass

    # Clean up outer archive if needed
    print("Archive extraction complete.")


def ensure_dataset(raw_data_dir=RAW_DATA_DIR, zip_path=DATASET_ZIP_PATH) -> str:
    """
    Ensure the UCI HAR dataset is downloaded and extracted.
    Returns path to raw_data_dir.
    """
    if is_dataset_ready(raw_data_dir):
        print(f"[OK] UCI HAR Dataset already exists and verified at: {raw_data_dir}")
        return raw_data_dir

    print("UCI HAR dataset not found or incomplete. Starting download...")
    os.makedirs(DATA_DIR, exist_ok=True)

    download_success = False
    for url in UCI_HAR_ZIP_URLS:
        print(f"Attempting download from: {url}")
        if download_file(url, zip_path):
            download_success = True
            break

    if not download_success:
        raise RuntimeError(
            "Failed to download UCI HAR Dataset from all known mirror URLs. "
            "Please check internet connection or manually place 'UCI HAR Dataset' in data/."
        )

    # Extract
    extract_zip(zip_path, DATA_DIR)

    # Some versions extract directly into DATA_DIR / 'UCI HAR Dataset'
    # Some versions might extract into DATA_DIR / 'UCI_HAR_Dataset'
    if not is_dataset_ready(raw_data_dir):
        # Check alternative folder names
        alt_names = ["UCI_HAR_Dataset", "UCI HAR Dataset"]
        for alt in alt_names:
            alt_path = os.path.join(DATA_DIR, alt)
            if is_dataset_ready(alt_path) and alt_path != raw_data_dir:
                # Rename or link
                os.rename(alt_path, raw_data_dir)
                break

    if not is_dataset_ready(raw_data_dir):
        raise RuntimeError(
            f"Extraction succeeded but directory structure verification failed in {raw_data_dir}."
        )

    print(f"[OK] UCI HAR Dataset successfully downloaded, extracted, and verified at: {raw_data_dir}")
    return raw_data_dir


if __name__ == "__main__":
    ensure_dataset()

