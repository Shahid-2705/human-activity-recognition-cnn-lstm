"""
Hybrid 1D CNN-LSTM Deep Learning Architecture for Human Activity Recognition.
Combines 1D Convolutional Neural Networks for local spatial/frequency feature extraction
with Long Short-Term Memory (LSTM) recurrent networks for temporal sequence modeling.
"""

import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import (
    NUM_CHANNELS,
    NUM_CLASSES,
    SEQ_LEN,
    get_device
)


class HAR_CNN_LSTM(nn.Module):
    """
    Hybrid 1D CNN-LSTM Architecture for Multichannel Sensor Time-Series Classification.
    
    Pipeline:
      1. Input: (Batch, in_channels=9, seq_len=128)
      2. CNN Block 1: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout
      3. CNN Block 2: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout
      4. Sequence Permutation: (Batch, channels, time_steps) -> (Batch, time_steps, channels)
      5. LSTM Recurrent Layers: Multi-layer bidirectional or unidirectional LSTM
      6. Temporal Aggregation: Last sequence state + mean sequence pooling
      7. Dense Classification Head: Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> Logits
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        num_classes: int = NUM_CLASSES,
        conv1_filters: int = 64,
        conv2_filters: int = 128,
        kernel_size: int = 5,
        pool_size: int = 2,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        bidirectional: bool = False,
        fc_hidden_dim: int = 64,
        dropout_rate: float = 0.3
    ):
        super(HAR_CNN_LSTM, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        padding = kernel_size // 2

        # 1D CNN Block 1: Low-level sensor feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv1_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.drop1 = nn.Dropout(dropout_rate)

        # 1D CNN Block 2: High-level hierarchical temporal features
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.drop2 = nn.Dropout(dropout_rate)

        # LSTM Layer for temporal sequence dependencies
        self.lstm = nn.LSTM(
            input_size=conv2_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_num_layers > 1 else 0.0
        )

        # Dimensions coming out of LSTM
        lstm_direction_factor = 2 if bidirectional else 1
        lstm_effective_dim = lstm_hidden_dim * lstm_direction_factor

        # Dense Classification Head
        self.fc1 = nn.Linear(lstm_effective_dim, fc_hidden_dim)
        self.ln = nn.LayerNorm(fc_hidden_dim)
        self.relu = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(fc_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, in_channels, seq_len)
               e.g. (Batch, 9, 128)
        Returns:
            logits: Tensor of shape (Batch, num_classes)
        """
        # CNN Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # CNN Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Reshape for LSTM: (Batch, channels, time_steps) -> (Batch, time_steps, channels)
        x = x.permute(0, 2, 1)

        # LSTM Temporal Modeling
        lstm_out, _ = self.lstm(x)  # shape: (Batch, seq_len_reduced, lstm_effective_dim)

        # Temporal representation: last time step representation
        temporal_feat = lstm_out[:, -1, :]

        # Dense Classification
        feat = self.fc1(temporal_feat)
        feat = self.ln(feat)
        feat = self.relu(feat)
        feat = self.drop3(feat)
        logits = self.classifier(feat)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Softmax probabilities for input sensor sequence."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict activity class index (0 to 5) for input sensor sequence."""
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=-1)


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    filepath: str,
    hyperparameters: Dict[str, Any] = None
) -> None:
    """Save full training checkpoint with model weights, optimizer, and metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "hyperparameters": hyperparameters or {}
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(
    filepath: str,
    device: torch.device = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a trained model and checkpoint metadata from disk.
    Automatically instantiates HAR_CNN_LSTM with the saved hyperparameters.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    hyperparams = checkpoint.get("hyperparameters", {})
    model = HAR_CNN_LSTM(
        in_channels=hyperparams.get("in_channels", NUM_CHANNELS),
        num_classes=hyperparams.get("num_classes", NUM_CLASSES),
        conv1_filters=hyperparams.get("conv1_filters", 64),
        conv2_filters=hyperparams.get("conv2_filters", 128),
        kernel_size=hyperparams.get("kernel_size", 5),
        pool_size=hyperparams.get("pool_size", 2),
        lstm_hidden_dim=hyperparams.get("lstm_hidden_dim", 128),
        lstm_num_layers=hyperparams.get("lstm_num_layers", 2),
        bidirectional=hyperparams.get("bidirectional", False),
        fc_hidden_dim=hyperparams.get("fc_hidden_dim", 64),
        dropout_rate=hyperparams.get("dropout_rate", 0.3)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


if __name__ == "__main__":
    device = get_device()
    test_model = HAR_CNN_LSTM().to(device)
    test_input = torch.randn(4, 9, 128).to(device)
    output = test_model(test_input)
    proba = test_model.predict_proba(test_input)
    preds = test_model.predict(test_input)
    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in test_model.parameters()):,}")
    print(f"Forward output logits shape: {output.shape}")
    print(f"Softmax probabilities shape: {proba.shape}")
    print(f"Predicted classes: {preds.tolist()}")

