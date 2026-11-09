"""
Visualization module for Human Activity Recognition.
Generates publication-quality Matplotlib/Seaborn figures for reports and
Plotly interactive charts for the Streamlit dashboard.
"""

import os
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import (
    ACTIVITY_NAMES,
    ACTIVITY_COLORS,
    INERTIAL_SIGNAL_NAMES,
    OUTPUTS_DIR
)


# Global styling for Matplotlib dark cyberpunk theme
plt.style.use("dark_background")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = None
) -> plt.Figure:
    """
    Plot Training and Validation Loss and Accuracy curves using Matplotlib.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0e1117")

    # 1. Loss Curve
    ax_loss = axes[0]
    ax_loss.set_facecolor("#161b22")
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="#38bdf8", linewidth=2.5, marker="o", markersize=4)
    ax_loss.plot(epochs, history["val_loss"], label="Val Loss", color="#f43f5e", linewidth=2.5, linestyle="--", marker="s", markersize=4)
    ax_loss.set_title("Cross-Entropy Loss vs. Epochs", fontsize=14, color="#f1f5f9", pad=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=12, color="#94a3b8")
    ax_loss.set_ylabel("Loss", fontsize=12, color="#94a3b8")
    ax_loss.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax_loss.legend(frameon=True, facecolor="#0e1117", edgecolor="#334155", fontsize=11)

    # 2. Accuracy Curve
    ax_acc = axes[1]
    ax_acc.set_facecolor("#161b22")
    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#818cf8", linewidth=2.5, marker="o", markersize=4)
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#10b981", linewidth=2.5, linestyle="--", marker="^", markersize=4)
    ax_acc.set_title("Classification Accuracy (%) vs. Epochs", fontsize=14, color="#f1f5f9", pad=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=12, color="#94a3b8")
    ax_acc.set_ylabel("Accuracy (%)", fontsize=12, color="#94a3b8")
    ax_acc.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax_acc.legend(frameon=True, facecolor="#0e1117", edgecolor="#334155", fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved training curves figure to: {save_path}")

    return fig


def plot_interactive_training_curves(history: Dict[str, List[float]]) -> go.Figure:
    """
    Generate an interactive dual-panel Plotly figure for training & validation metrics.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cross-Entropy Loss Curve", "Classification Accuracy Curve (%)"),
        horizontal_spacing=0.12
    )

    # Loss subplot
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=5)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["val_loss"],
            mode="lines+markers",
            name="Val Loss",
            line=dict(color="#f43f5e", width=3, dash="dot"),
            marker=dict(size=5)
        ),
        row=1, col=1
    )

    # Accuracy subplot
    fig.add_trace(
        go.Scatter(
            x=epochs, y=[a * 100 for a in history["train_acc"]],
            mode="lines+markers",
            name="Train Acc (%)",
            line=dict(color="#818cf8", width=3),
            marker=dict(size=5)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=[a * 100 for a in history["val_acc"]],
            mode="lines+markers",
            name="Val Acc (%)",
            line=dict(color="#10b981", width=3, dash="dot"),
            marker=dict(size=5)
        ),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", color="#f1f5f9"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(14, 17, 23, 0.8)",
            bordercolor="#334155",
            borderwidth=1
        )
    )

    fig.update_xaxes(gridcolor="#262f3d", title_text="Epoch")
    fig.update_yaxes(gridcolor="#262f3d", row=1, col=1, title_text="Loss")
    fig.update_yaxes(gridcolor="#262f3d", row=1, col=2, title_text="Accuracy (%)")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ACTIVITY_NAMES,
    save_path: str = None
) -> plt.Figure:
    """
    Plot normalized confusion matrix using Seaborn heatmap.
    """
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#0e1117"
    )

    ax.set_title("Normalized Confusion Matrix", fontsize=15, color="#f1f5f9", pad=14, fontweight="bold")
    ax.set_xlabel("Predicted Activity", fontsize=12, color="#94a3b8", labelpad=10)
    ax.set_ylabel("True Activity", fontsize=12, color="#94a3b8", labelpad=10)
    plt.xticks(rotation=45, ha="right", color="#e2e8f0")
    plt.yticks(rotation=0, color="#e2e8f0")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved confusion matrix figure to: {save_path}")

    return fig


def plot_interactive_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ACTIVITY_NAMES
) -> go.Figure:
    """
    Generate an interactive Plotly heatmap for the confusion matrix.
    """
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    text_labels = [[f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(len(class_names))] for i in range(len(class_names))]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=class_names,
            y=class_names,
            text=text_labels,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#0e1117"],
                [0.3, "#1e3a8a"],
                [0.7, "#2563eb"],
                [1.0, "#38bdf8"]
            ],
            showscale=True,
            colorbar=dict(title="Ratio", tickformat=".0%")
        )
    )

    fig.update_layout(
        title=dict(text="Interactive Confusion Matrix (UCI HAR Test Set)", font=dict(size=16, color="#f8fafc")),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        xaxis=dict(title="Predicted Activity", tickangle=-35),
        yaxis=dict(title="True Ground-Truth Activity", autorange="reversed"),
        font=dict(family="Inter, sans-serif", color="#f1f5f9"),
        margin=dict(l=60, r=40, t=60, b=80)
    )

    return fig


def plot_class_performance_bar(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive grouped bar chart of Precision, Recall, and F1-Score per class.
    """
    report = metrics.get("classification_report", {})
    categories = [cat for cat in ACTIVITY_NAMES if cat in report]

    precisions = [report[c]["precision"] * 100 for c in categories]
    recalls = [report[c]["recall"] * 100 for c in categories]
    f1_scores = [report[c]["f1-score"] * 100 for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=precisions, name="Precision (%)", marker_color="#38bdf8"))
    fig.add_trace(go.Bar(x=categories, y=recalls, name="Recall (%)", marker_color="#818cf8"))
    fig.add_trace(go.Bar(x=categories, y=f1_scores, name="F1-Score (%)", marker_color="#10b981"))

    fig.update_layout(
        barmode="group",
        title="Per-Class Performance Metrics (Precision, Recall, F1)",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", color="#f1f5f9"),
        yaxis=dict(title="Percentage (%)", range=[60, 105], gridcolor="#262f3d"),
        xaxis=dict(gridcolor="#262f3d"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    return fig


def plot_sensor_signals(
    signals_window: np.ndarray,
    predicted_activity: str = None,
    confidence: float = None
) -> go.Figure:
    """
    Plot interactive 9-channel (or 6-channel) sensor waveforms.
    signals_window shape: (128, 9) or (9, 128)
    Columns: body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z
    """
    if signals_window.shape[0] == 9 and signals_window.shape[1] > 9:
        signals_window = signals_window.T  # Transpose to (time_steps, 9)

    time_steps = np.arange(signals_window.shape[0])
    time_seconds = time_steps / 50.0  # 50 Hz sampling rate

    title_text = "Multi-Channel Inertial Sensor Waveforms (50 Hz Smartphone IMU)"
    if predicted_activity:
        conf_str = f" ({confidence:.1f}% confidence)" if confidence is not None else ""
        title_text += f"<br><sup>Predicted Activity: <b style='color:#38bdf8'>{predicted_activity}</b>{conf_str}</sup>"

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Triaxial Body Acceleration (g)",
            "Triaxial Angular Velocity / Gyroscope (rad/s)",
            "Triaxial Total Acceleration (g)"
        )
    )

    # 1. Body Acceleration
    fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 0], mode="lines", name="Body Acc X", line=dict(color="#38bdf8", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 1], mode="lines", name="Body Acc Y", line=dict(color="#818cf8", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 2], mode="lines", name="Body Acc Z", line=dict(color="#ec4899", width=2)), row=1, col=1)

    # 2. Body Gyroscope
    fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 3], mode="lines", name="Body Gyro X", line=dict(color="#fbbf24", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 4], mode="lines", name="Body Gyro Y", line=dict(color="#34d399", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 5], mode="lines", name="Body Gyro Z", line=dict(color="#a78bfa", width=2)), row=2, col=1)

    # 3. Total Acceleration
    if signals_window.shape[1] >= 9:
        fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 6], mode="lines", name="Total Acc X", line=dict(color="#f97316", width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 7], mode="lines", name="Total Acc Y", line=dict(color="#06b6d4", width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=time_seconds, y=signals_window[:, 8], mode="lines", name="Total Acc Z", line=dict(color="#e11d48", width=2)), row=3, col=1)

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color="#f8fafc")),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", color="#f1f5f9"),
        height=650,
        hovermode="x unified",
        margin=dict(l=50, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1, gridcolor="#262f3d")
    fig.update_yaxes(gridcolor="#262f3d")

    return fig


def plot_probability_distribution(probabilities: np.ndarray, predicted_class_idx: int = None) -> go.Figure:
    """
    Generate horizontal bar chart showing Softmax confidence across all 6 activity classes.
    """
    probs_pct = probabilities * 100
    colors = [ACTIVITY_COLORS.get(name, "#38bdf8") for name in ACTIVITY_NAMES]

    # Add border/highlight to top prediction
    marker_lines = [dict(width=2, color="#ffffff") if i == predicted_class_idx else dict(width=0) for i in range(len(ACTIVITY_NAMES))]

    fig = go.Figure(
        go.Bar(
            x=probs_pct,
            y=ACTIVITY_NAMES,
            orientation="h",
            marker=dict(
                color=colors,
                line=marker_lines[0] if len(marker_lines) > 0 else None
            ),
            text=[f"{p:.1f}%" for p in probs_pct],
            textposition="auto",
            textfont=dict(color="#ffffff", size=13, family="Inter, sans-serif")
        )
    )

    fig.update_layout(
        title=dict(text="Class Probability Distribution", font=dict(size=16, color="#f8fafc")),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", color="#f1f5f9"),
        xaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="#262f3d"),
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig

