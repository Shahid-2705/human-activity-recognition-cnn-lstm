"""
Evaluation module for Human Activity Recognition.
Computes test metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix,
and per-class classification reports.
"""

import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import (
    ACTIVITY_NAMES,
    BEST_MODEL_PATH,
    METRICS_JSON_PATH,
    CONFUSION_MATRIX_PATH,
    get_device
)
from models.cnn_lstm import load_model_checkpoint
from preprocessing.data_loader import get_data_loaders
from visualization.plots import plot_confusion_matrix


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = None
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluate PyTorch model on a DataLoader.
    Computes loss, accuracy, precision, recall, F1, and confusion matrix.
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds)
    cls_report = classification_report(
        all_targets,
        all_preds,
        target_names=ACTIVITY_NAMES,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "test_loss": float(avg_loss),
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report
    }

    return metrics, cm


def run_evaluation(checkpoint_path: str = BEST_MODEL_PATH):
    """Load model checkpoint, evaluate on official test split, and save results."""
    device = get_device()
    print(f"Using compute device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}. Train the model first.")

    model, checkpoint_info = load_model_checkpoint(checkpoint_path, device=device)
    print(f"Loaded checkpoint trained for {checkpoint_info.get('epoch', '?')} epochs.")

    _, _, test_loader, _ = get_data_loaders()

    print("\nRunning comprehensive model evaluation on official UCI HAR Test Set...")
    metrics, cm = evaluate_model(model, test_loader, device=device)

    print("\n" + "=" * 55)
    print("           MODEL TEST EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Test Accuracy:       {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:     {metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall:        {metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score:      {metrics['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1-Score:   {metrics['f1_weighted'] * 100:.2f}%")
    print(f"Test Loss:           {metrics['test_loss']:.4f}")
    print("=" * 55)

    print("\nClassification Report (Per-Activity Metrics):")
    cls_dict = metrics["classification_report"]
    print(f"{'Activity':<24}{'Precision (%)':<16}{'Recall (%)':<14}{'F1-Score (%)':<14}{'Support':<10}")
    print("-" * 75)
    for name in ACTIVITY_NAMES:
        if name in cls_dict:
            p = cls_dict[name]["precision"] * 100
            r = cls_dict[name]["recall"] * 100
            f1 = cls_dict[name]["f1-score"] * 100
            supp = cls_dict[name]["support"]
            print(f"{name:<24}{p:<16.2f}{r:<14.2f}{f1:<14.2f}{supp:<10}")
    print("-" * 75)

    # Save metrics JSON
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved evaluation metrics to: {METRICS_JSON_PATH}")

    # Plot & Save Confusion Matrix
    plot_confusion_matrix(cm, ACTIVITY_NAMES, save_path=CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix plot to: {CONFUSION_MATRIX_PATH}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
