# Human Activity Recognition Using Hybrid 1D CNN-LSTM

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20GPU-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning application for **Human Activity Recognition (HAR)** using triaxial accelerometer and gyroscope time-series signals from smartphones. The architecture combines **1D Convolutional Neural Networks (1D-CNN)** for invariant local motion extraction with **Long Short-Term Memory (LSTM)** recurrent networks for temporal sequence learning.

---

## 1. Project Overview

Human Activity Recognition (HAR) enables context-aware computing, elderly fall detection, sports analytics, and health monitoring. In this project, raw inertial telemetry collected by a waist-mounted smartphone (Samsung Galaxy S II) is classified into six activities:

| Activity ID | Class Name | Motion Category | Biomechanical Characteristics |
|:---:|:---|:---|:---|
| **0** | **Walking** | Dynamic | Low-frequency periodic oscillations across triaxial accelerometer channels |
| **1** | **Walking Upstairs** | Dynamic | High upward acceleration impact peaks and periodic rhythmic pitch |
| **2** | **Walking Downstairs** | Dynamic | Sharp deceleration impact spikes with downward gravitational alignment |
| **3** | **Sitting** | Static | Low signal variance; gravity vector dominant along the device lateral axis |
| **4** | **Standing** | Static | Low signal variance; gravity vector dominant vertically with subtle postural sway |
| **5** | **Laying** | Static | Gravity vector rotated orthogonally across body-relative axes |

---

## 2. Dataset: UCI HAR Dataset

- **Source**: [UCI Machine Learning Repository #240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Subjects**: 30 volunteers aged 19–48 years performing daily living activities.
- **Sensors**: Triaxial Accelerometer and Gyroscope sampled at **50 Hz** (20 ms interval).
- **Segmentation**: Sliding windows of **128 readings (2.56 seconds)** with **50% overlap** (64 samples step).
- **Official Split**:
  - **Training**: 7,352 sequence windows (70%)
  - **Testing**: 2,947 sequence windows (30%)

### Raw Inertial Signals vs. 561 Handcrafted Features
While the UCI HAR dataset provides 561 pre-engineered statistical summary features (means, standard deviations, FFT energy bands), flattening them into an artificial matrix destroys spatial-temporal continuity. This project reconstructs and directly feeds the **genuine 9-channel raw inertial time series**:
1. `body_acc_x`, `body_acc_y`, `body_acc_z` (Triaxial Body Acceleration in $g$)
2. `body_gyro_x`, `body_gyro_y`, `body_gyro_z` (Triaxial Angular Velocity in $\text{rad/s}$)
3. `total_acc_x`, `total_acc_y`, `total_acc_z` (Triaxial Total Acceleration including gravity in $g$)

---

## 3. Deep Learning Architecture: Hybrid 1D CNN-LSTM

```
                        [ Input Sensor Window: (Batch, 9, 128) ]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │  1D CNN Block 1: Conv1D (64, k=5, pad=2)    │
                   │  BatchNorm1d + ReLU + MaxPool1d (stride=2)  │
                   │  Dropout (p=0.3)                            │
                   └──────────────────────┬──────────────────────┘
                                          ▼ Shape: (Batch, 64, 64)
                   ┌─────────────────────────────────────────────┐
                   │  1D CNN Block 2: Conv1D (128, k=5, pad=2)   │
                   │  BatchNorm1d + ReLU + MaxPool1d (stride=2)  │
                   │  Dropout (p=0.3)                            │
                   └──────────────────────┬──────────────────────┘
                                          ▼ Shape: (Batch, 128, 32)
                   ┌─────────────────────────────────────────────┐
                   │  Permute to Sequence: (Batch, 32, 128)      │
                   └──────────────────────┬──────────────────────┘
                                          ▼
                   ┌─────────────────────────────────────────────┐
                   │  2-Layer LSTM (Hidden=128, BatchFirst=True) │
                   │  Dropout (p=0.3)                            │
                   └──────────────────────┬──────────────────────┘
                                          ▼ Shape: (Batch, 32, 128)
                   ┌─────────────────────────────────────────────┐
                   │  Temporal Extraction: Last Sequence Step    │
                   └──────────────────────┬──────────────────────┘
                                          ▼ Shape: (Batch, 128)
                   ┌─────────────────────────────────────────────┐
                   │  Dense FC Head: Linear(128 -> 64)           │
                   │  LayerNorm(64) + ReLU + Dropout(0.3)        │
                   │  Linear(64 -> 6) (Logits)                   │
                   └──────────────────────┬──────────────────────┘
                                          ▼
                          [ Softmax Output Probabilities ]
```

### Architectural Highlights
- **1D Convolutions**: Function as learnable finite impulse response (FIR) filter banks that extract invariant movement micro-patterns without manual feature engineering.
- **LSTM Recurrent Layers**: Model chronological gait phase transitions, step periodicity, and sustained postures across consecutive time steps.
- **Normalization Strategy**: Channel-wise standardization fitted strictly on the training set to prevent data leakage.
- **Regularization**: Batch Normalization, Layer Normalization, Dropout ($p=0.3$), and Weight Decay ($10^{-4}$).

---

## 4. Empirical Evaluation Results

Evaluated on the **official independent UCI HAR test split (2,947 samples)**:

| Metric | Score |
|:---|:---:|
| **Overall Test Accuracy** | **92.77%** |
| **Macro Precision** | **92.92%** |
| **Macro Recall** | **93.05%** |
| **Macro F1-Score** | **92.97%** |
| **Weighted F1-Score** | **92.75%** |
| **Test Cross-Entropy Loss** | **0.1986** |

### Per-Activity Breakdown

| Activity | Precision (%) | Recall (%) | F1-Score (%) | Test Support |
|:---|:---:|:---:|:---:|:---:|
| **Walking** | 100.00% | 99.40% | **99.70%** | 496 |
| **Walking Upstairs** | 97.89% | 98.30% | **98.09%** | 471 |
| **Walking Downstairs** | 97.67% | 100.00% | **98.82%** | 420 |
| **Sitting** | 78.39% | 81.26% | **79.80%** | 491 |
| **Standing** | 83.56% | 79.32% | **81.39%** | 532 |
| **Laying** | 100.00% | 100.00% | **100.00%** | 537 |

*Note: Dynamic activities (Walking, Upstairs, Downstairs) and Laying achieve near-perfect discrimination (>98%–100%). Sitting and Standing exhibit slight mutual confusion due to identical static gravitational orientation, matching established academic literature.*

---

## 5. Project Structure

```
human_activity_recognition/
│
├── main.py                    # One command to run everything: python main.py
├── api.py                     # FastAPI wrapper around the CNN-LSTM sensor model
├── web/                       # Node (Express) UI
│   ├── server.js              #   serves the UI, relays phone-camera streams, proxies /api
│   ├── lib/phone-url.js       #   only local-network phone-camera addresses are accepted
│   ├── public/                #   index.html, style.css and js/ (app, pose-classifier, knn)
│   ├── scripts/download-models.js  # fetches the MediaPipe pose / face / hand models
│   └── test/                  #   44 unit tests (classifier, hand actions, calibration, URL rules)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and benchmarks
│
├── data/
│   ├── __init__.py
│   ├── download_dataset.py    # Auto-downloads and extracts UCI HAR dataset
│   └── UCI HAR Dataset/       # Extracted dataset directory
│
├── models/
│   ├── __init__.py
│   └── cnn_lstm.py            # PyTorch 1D CNN-LSTM architecture & checkpoint loaders
│
├── training/
│   ├── __init__.py
│   └── train.py               # Complete training pipeline with early stopping & scheduler
│
├── preprocessing/
│   ├── __init__.py
│   └── data_loader.py         # 9-channel inertial sequence parser & SensorScaler
│
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py            # Comprehensive evaluation script & metrics generation
│
├── visualization/
│   ├── __init__.py
│   └── plots.py               # Plotly interactive graphs & Matplotlib figures
│
├── utils/
│   ├── __init__.py
│   └── config.py              # Centralized configuration, hyperparameters & paths
│
├── saved_models/
│   ├── best_cnn_lstm.pt       # Checkpoint with highest validation accuracy
│   ├── final_cnn_lstm.pt      # Final training checkpoint
│   └── scaler_params.json     # Fitted channel standardization parameters
│
├── outputs/
│   ├── evaluation_metrics.json# Test split performance metrics
│   ├── training_history.json  # Loss & accuracy trajectories across epochs
│   ├── confusion_matrix.png   # Publication-quality confusion matrix
│   └── loss_accuracy_curves.png # Training vs. validation curves
│
└── sample_data/               # Pre-extracted 128x9 sample CSVs for live UI testing
    ├── sample_walking.csv
    ├── sample_walking_upstairs.csv
    ├── sample_walking_downstairs.csv
    ├── sample_sitting.csv
    ├── sample_standing.csv
    ├── sample_laying.csv
    └── sample_multi_window_recording.csv
```

---

## 6. Installation & Execution Guide

### Step 1: Clone or Navigate to the Repository
```bash
cd d:\dl\human_activity_recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset (Optional - Handled Automatically)
```bash
python data/download_dataset.py
```

### Step 4: Train the CNN-LSTM Model
```bash
python training/train.py
```
*Note: Uses CUDA GPU acceleration automatically if an NVIDIA GPU is detected, with automatic CPU fallback.*

### Step 5: Run Standalone Evaluation
```bash
python evaluation/evaluate.py
```

### Step 6: Run the app (one command)
```bash
python main.py
```
It runs, in order: `npm install` (first run only), the Node UI on **http://localhost:3000**, then the Python sensor API on port 8000. Both print into the same terminal (`[node]` / `[python]`) and `Ctrl+C` stops both. Your browser opens when it is ready; add `--no-browser` to skip that.

Requirements: Python 3.11+ and Node 18+. The first `npm install` also downloads the MediaPipe pose, face and hand-gesture models (about 21 MB) into `web/public/models/`.

Options (environment variables): `PORT` (UI, default 3000), `API_PORT` (Python API, default 8000), `HOST` (default `127.0.0.1`, set `0.0.0.0` to open the dashboard to your network).

Manual alternative: `cd web && npm install && npm start` (this also starts the Python API; set `START_API=0` to run the UI alone).

---

## 7. Web UI

A compact two-tab interface:
- **Live camera**: pose and face models run **in your browser** (nothing is uploaded) and detect
  - *Activity*: Standing, Sitting, Laying, Walking, Running, Jumping, Bending (or "Upper body only" when hips/knees are out of view)
  - *Gesture*: Waving, Hands Up, Hand Raised, Clapping, Eating / Drinking, Phone Call, Hands together, Hand at face, plus finger signs (Thumbs up/down, Peace sign, Pointing up, I love you, Fist, Open palm)
  - *Expression* (estimate): Crying-like, Smiling, Neutral
  - Continuous capture in 30 s - 5 min segments with a per-segment time breakdown.
  - Dashboard cards: session stats (running time, frame rate, camera, longest streak, activity changes), a 5-minute activity timeline, finished-segment history, and a guide to what can be detected.
  - **Train on my own activities**: record 10 s clips of any activity (crying, yawning, typing, ...); a kNN model learns from them and is stored in your browser's localStorage.
- **Sensor model**: runs the trained 1D CNN-LSTM on the recorded smartphone sensor samples and shows class probabilities.

**Sitting vs standing**: leg geometry is unreliable on a webcam (legs are usually cropped), so press **Calibrate** once. It records 10 s sitting and 10 s standing and learns from body position only (size and height in the frame, which body parts are visible), ignoring your arms. On a recording of one user, a model trained on just those two clips called every unseen standing clip "Standing".

**Clapping** is the weakest gesture: the hand model drops a hand whenever the palms meet, so a clap is only caught about 40-60% of the time. **Eating / drinking** and **phone call** use the hand model's fingertips and palms against the face.

The live camera uses MediaPipe body, hand and face landmarks with measured rules. It is a practical estimate, not a clinical or safety-grade measurement. "Crying-like" only reflects facial expression, walking needs about 1-2 seconds of movement before it registers, and thresholds were tuned on recordings of one person in one room, so accuracy in your setup may differ. The **Train on my own activities** panel and **Calibrate** exist so you can tune it to your own camera.

Run the tests with `cd web && npm test` (44 tests).

---

## 8. Choosing a camera: webcam or phone

Pick **Webcam** or **Phone camera** at the top of the Live tab. Your choice (and the phone address) is remembered.

### Phone camera over Wi-Fi (DroidCam / IP Webcam) - no PC software needed
1. Put the phone and the PC on the **same Wi-Fi** (or connect the PC to the phone's hotspot).
2. Open **DroidCam** on the phone and leave it open and awake. It shows a **WiFi IP** and a **port** (usually `4747`), e.g. `192.168.1.105`.
3. Check it on the PC first: open `http://192.168.1.105:4747/video` (use your own IP) in a browser. If you see live video, the phone is reachable.
4. Run `python main.py`, choose **Phone camera**, paste that address into the box and press **Start**.

IP Webcam works the same way; use the video address it shows (for example `http://192.168.1.105:8080/video`).

### Webcam (PC camera, USB camera, or phone via the DroidCam desktop client)
Install the **DroidCam Client** on Windows, connect it to the phone (Wi-Fi or USB) and click Start. The phone then appears as **DroidCam Video** in the camera drop-down; stay on **Webcam** and pick it. Iriun and Camo work the same way. You can also load a video file.

Use only one of the two DroidCam routes at a time: the free app generally accepts a single connection.

### Troubleshooting
- **"Could not reach the phone camera"**: check both devices are on the same network (guest Wi-Fi often isolates devices), the phone app is open, and no VPN or firewall is blocking it. Some versions use `/mjpegfeed` instead of `/video`.
- **"Use an http(s) address on your local network"**: only `192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`, `*.local` and `localhost` addresses are accepted; internet addresses and addresses containing a username/password are refused on purpose.
- **"Device in use"** (webcam): only one program can use a webcam at a time. Close other tabs or apps using it (Teams, Zoom, the Camera app, another copy of this page).
- **Best results**: phone in landscape, about 2 m away so your head to knees are in view, then press **Calibrate** once.

### Privacy
Video is processed in your browser; nothing is uploaded. A phone stream is relayed by the local Node server only. Your recorded training samples and the sit/stand calibration are stored in your browser's localStorage. The dashboard listens on `127.0.0.1` only unless you set `HOST`.

---

## 9. Citation & Academic Reference

```bibtex
@inproceedings{anguita2013public,
  title={A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L},
  booktitle={21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year={2013}
}
```

