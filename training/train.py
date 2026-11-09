"""
Training pipeline for Human Activity Recognition using Hybrid 1D CNN-LSTM.
Includes stratified validation, early stopping, ReduceLROnPlateau scheduling,
checkpoint management, and automatic test evaluation.
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import (
    NUM_CHANNELS,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    RANDOM_SEED,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    TRAIN_HISTORY_PATH,
    ROC_CURVES_PATH,
    METRICS_JSON_PATH,
    CONFUSION_MATRIX_PATH,
    ACTIVITY_NAMES,
    get_device
)
from preprocessing.data_loader import get_data_loaders
from models.cnn_lstm import (
    HAR_CNN_LSTM,
    save_model_checkpoint
)
from evaluation.evaluate import evaluate_model
from visualization.plots import (
    plot_training_curves,
    plot_confusion_matrix
)


def set_seed(seed: int = RANDOM_SEED):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping to stabilize LSTM gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_har_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    conv1_filters: int = 64,
    conv2_filters: int = 128,
    lstm_hidden_dim: int = 128,
    lstm_num_layers: int = 2,
    dropout_rate: float = 0.3
):
    """Complete training and evaluation pipeline."""
    set_seed(RANDOM_SEED)
    device = get_device()

    print("=" * 65)
    print("      HUMAN ACTIVITY RECOGNITION (HAR) - CNN-LSTM TRAINING")
    print("=" * 65)
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Hyperparameters: Batch Size={batch_size}, LR={learning_rate}, Epochs={epochs}, Patience={patience}")

    # Load data
    train_loader, val_loader, test_loader, scaler = get_data_loaders(batch_size=batch_size)

    # Instantiate Model
    hyperparams = {
        "in_channels": NUM_CHANNELS,
        "num_classes": NUM_CLASSES,
        "conv1_filters": conv1_filters,
        "conv2_filters": conv2_filters,
        "kernel_size": 5,
        "pool_size": 2,
        "lstm_hidden_dim": lstm_hidden_dim,
        "lstm_num_layers": lstm_num_layers,
        "bidirectional": False,
        "fc_hidden_dim": 64,
        "dropout_rate": dropout_rate
    }

    model = HAR_CNN_LSTM(**hyperparams).to(device)
    print(f"Model Architecture:\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<14}{'Time (s)':<10}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(curr_lr))

        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc*100:<16.2f}{val_loss:<14.4f}{val_acc*100:<14.2f}{epoch_time:<10.1f}")

        # Check for best validation accuracy or loss
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_acc": val_acc},
                filepath=BEST_MODEL_PATH,
                hyperparameters=hyperparams
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} consecutive epochs. Stopping at epoch {epoch}.")
                break

    total_training_time = time.time() - start_time
    print("-" * 75)
    print(f"Training completed in {total_training_time:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}% (Loss: {best_val_loss:.4f})")

    # Save final model
    save_model_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=len(history["train_loss"]),
        metrics={"final_val_loss": history["val_loss"][-1], "final_val_acc": history["val_acc"][-1]},
        filepath=FINAL_MODEL_PATH,
        hyperparameters=hyperparams
    )

    # Save training history JSON
    with open(TRAIN_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {TRAIN_HISTORY_PATH}")

    # Plot and save training curves
    plot_training_curves(history, save_path=ROC_CURVES_PATH)

    # Evaluate Best Model on the official Test Split
    print("\n" + "=" * 65)
    print("          EVALUATING BEST CHECKPOINT ON TEST SPLIT")
    print("=" * 65)
    from models.cnn_lstm import load_model_checkpoint
    best_model, _ = load_model_checkpoint(BEST_MODEL_PATH, device=device)
    test_metrics, cm = evaluate_model(best_model, test_loader, device=device)

    print(f"Test Accuracy:       {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:      {test_metrics['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1-Score:   {test_metrics['f1_weighted'] * 100:.2f}%")
    print(f"Test Loss:           {test_metrics['test_loss']:.4f}")

    # Save metrics JSON & Confusion Matrix image
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(test_metrics, f, indent=4)
    plot_confusion_matrix(cm, ACTIVITY_NAMES, save_path=CONFUSION_MATRIX_PATH)
    print("All training and evaluation artifacts saved successfully.")

    return test_metrics


if __name__ == "__main__":
    train_har_model()
