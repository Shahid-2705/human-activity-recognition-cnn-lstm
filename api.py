"""
Small JSON API around the trained 1D CNN-LSTM sensor model.
Used by the Node UI (web/) for the "Sensor Model" tab.

Run:  python -m uvicorn api:app --port 8000
"""

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.config import (  # noqa: E402
    ACTIVITY_NAMES,
    BEST_MODEL_PATH,
    OUTPUTS_DIR,
    SAMPLE_DATA_DIR,
    SAVED_MODELS_DIR,
    SEQ_LEN,
    INERTIAL_SIGNAL_NAMES,
    get_device,
)
from models.cnn_lstm import load_model_checkpoint  # noqa: E402
from preprocessing.data_loader import SensorScaler  # noqa: E402

app = FastAPI(title="HAR sensor model API")

device = get_device()
model, _ = load_model_checkpoint(BEST_MODEL_PATH, device=device)
model.eval()
scaler = SensorScaler()
scaler.load(os.path.join(SAVED_MODELS_DIR, "scaler_params.json"))


def _test_accuracy():
    path = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("accuracy")
    except (OSError, ValueError):
        return None


def _sample_names():
    return sorted(
        f[:-4] for f in os.listdir(SAMPLE_DATA_DIR)
        if f.startswith("sample_") and f.endswith(".csv") and "multi_window" not in f
    )


class PredictRequest(BaseModel):
    sample: str | None = None            # e.g. "sample_walking"
    window: list[list[float]] | None = None  # (128, 9)


@app.get("/api/status")
def status():
    return {
        "device": str(device),
        "classes": ACTIVITY_NAMES,
        "test_accuracy": _test_accuracy(),
        "channels": INERTIAL_SIGNAL_NAMES,
    }


@app.get("/api/samples")
def samples():
    return {"samples": _sample_names()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.sample:
        if not re.fullmatch(r"sample_[a-z_]+", req.sample) or req.sample not in _sample_names():
            raise HTTPException(404, "Unknown sample")
        df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, req.sample + ".csv"))
        window = df[INERTIAL_SIGNAL_NAMES].to_numpy(dtype=np.float32)
    elif req.window:
        window = np.asarray(req.window, dtype=np.float32)
    else:
        raise HTTPException(400, "Provide 'sample' or 'window'")

    if window.ndim != 2 or window.shape[1] != len(INERTIAL_SIGNAL_NAMES) or window.shape[0] < SEQ_LEN:
        raise HTTPException(400, f"Window must be at least ({SEQ_LEN}, {len(INERTIAL_SIGNAL_NAMES)})")
    window = window[:SEQ_LEN]

    x = torch.tensor(scaler.transform(window.T[np.newaxis]), dtype=torch.float32).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = model.predict_proba(x).cpu().numpy().squeeze()
    latency_ms = (time.perf_counter() - t0) * 1000
    idx = int(np.argmax(probs))
    return {
        "prediction": ACTIVITY_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {name: float(p) for name, p in zip(ACTIVITY_NAMES, probs)},
        "latency_ms": round(latency_ms, 2),
    }
