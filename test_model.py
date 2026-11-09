"""
Standalone Model Inference and Testing Script.
Evaluates model on sample sensor recordings, verifies latency, and tests multi-window sequences.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.config import (
    BEST_MODEL_PATH,
    SAVED_MODELS_DIR,
    SAMPLE_DATA_DIR,
    ACTIVITY_LABEL_MAP,
    ACTIVITY_NAMES,
    get_device
)
from models.cnn_lstm import load_model_checkpoint
from preprocessing.data_loader import SensorScaler


def run_tests():
    device = get_device()
    print("=" * 80)
    print("           HUMAN ACTIVITY RECOGNITION - MODEL INFERENCE TEST")
    print("=" * 80)
    print(f"Compute Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model Path     : {BEST_MODEL_PATH}")

    # Load model and scaler
    model, checkpoint = load_model_checkpoint(BEST_MODEL_PATH, device=device)
    model.eval()

    scaler = SensorScaler()
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler_params.json")
    scaler.load(scaler_path)

    # GPU Warmup
    warmup_tensor = torch.randn(1, 9, 128, dtype=torch.float32).to(device)
    with torch.no_grad():
        _ = model.predict_proba(warmup_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 1. Test individual activity sample files
    test_cases = [
        ("Walking", "sample_walking.csv"),
        ("Walking Upstairs", "sample_walking_upstairs.csv"),
        ("Walking Downstairs", "sample_walking_downstairs.csv"),
        ("Sitting", "sample_sitting.csv"),
        ("Standing", "sample_standing.csv"),
        ("Laying", "sample_laying.csv"),
    ]

    print("\n--- 1. SINGLE-WINDOW BENCHMARK TESTS (128 samples / 2.56 sec @ 50Hz) ---")
    header = f"{'Ground Truth':<22} | {'Predicted Activity':<22} | {'Confidence':<12} | {'Latency':<12} | {'Result'}"
    print(header)
    print("-" * len(header))

    passed = 0
    total_latency = 0.0

    for ground_truth, filename in test_cases:
        filepath = os.path.join(SAMPLE_DATA_DIR, filename)
        df = pd.read_csv(filepath)
        raw_vals = df.values[:128, :9]
        inp = raw_vals.T[np.newaxis, :, :]  # Shape: (1, 9, 128)
        norm_inp = scaler.transform(inp)

        # Warmup and timed inference
        tensor_in = torch.tensor(norm_inp, dtype=torch.float32).to(device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = model.predict_proba(tensor_in).cpu().numpy().squeeze()
            pred_idx = int(np.argmax(probs))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        total_latency += latency

        pred_label = ACTIVITY_LABEL_MAP[pred_idx]
        confidence = probs[pred_idx] * 100
        match = (pred_label == ground_truth)
        if match:
            passed += 1
        tag = "[MATCH]" if match else "[MISMATCH]"

        print(f"{ground_truth:<22} | {pred_label:<22} | {confidence:>8.2f}%    | {latency:>7.2f} ms   | {tag}")

    avg_latency = total_latency / len(test_cases)
    print("-" * len(header))
    print(f"Benchmark Accuracy : {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Average Latency    : {avg_latency:.2f} ms per window")

    # 2. Test Continuous Multi-Window Stream (640 samples = 5 consecutive windows)
    print("\n--- 2. MULTI-WINDOW CONTINUOUS STREAM TEST ---")
    multi_file = os.path.join(SAMPLE_DATA_DIR, "sample_multi_window_recording.csv")
    if os.path.exists(multi_file):
        df_multi = pd.read_csv(multi_file)
        num_windows = df_multi.shape[0] // 128
        print(f"Loaded multi-window file ({df_multi.shape[0]} samples = {num_windows} consecutive windows)")

        print(f"{'Window #':<10} | {'Time Range':<16} | {'Predicted Activity':<24} | {'Confidence'}")
        print("-" * 65)

        for w in range(num_windows):
            chunk = df_multi.values[w*128:(w+1)*128, :9]
            inp_chunk = chunk.T[np.newaxis, :, :]
            norm_chunk = scaler.transform(inp_chunk)
            tensor_chunk = torch.tensor(norm_chunk, dtype=torch.float32).to(device)

            with torch.no_grad():
                probs_chunk = model.predict_proba(tensor_chunk).cpu().numpy().squeeze()
                chunk_pred_idx = int(np.argmax(probs_chunk))

            time_range = f"{w*2.56:.2f}s - {(w+1)*2.56:.2f}s"
            act_name = ACTIVITY_LABEL_MAP[chunk_pred_idx]
            conf = probs_chunk[chunk_pred_idx] * 100
            print(f"Window {w+1:<3} | {time_range:<16} | {act_name:<24} | {conf:.2f}%")

    print("\n" + "=" * 80)
    print("                    ALL INFERENCE TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
