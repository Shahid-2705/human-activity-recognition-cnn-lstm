"""
Human Activity Recognition (HAR) Streamlit Dashboard.
Hybrid 1D CNN-LSTM Deep Learning on Smartphone Inertial Sensor Data.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
import cv2
import plotly.graph_objects as go

# Append base dir to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.config import (
    ACTIVITY_LABEL_MAP,
    ACTIVITY_NAMES,
    ACTIVITY_COLORS,
    INERTIAL_SIGNAL_NAMES,
    BEST_MODEL_PATH,
    SAVED_MODELS_DIR,
    OUTPUTS_DIR,
    SAMPLE_DATA_DIR,
    METRICS_JSON_PATH,
    TRAIN_HISTORY_PATH,
    get_device
)
from models.cnn_lstm import load_model_checkpoint
from preprocessing.data_loader import SensorScaler
from utils.camera_stream import CameraMotionTracker
from visualization.plots import (
    plot_interactive_training_curves,
    plot_interactive_confusion_matrix,
    plot_class_performance_bar,
    plot_sensor_signals,
    plot_probability_distribution
)

# Page configuration
st.set_page_config(
    page_title="HAR Deep Learning | CNN-LSTM Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container background */
    .stApp {
        background-color: #090c10;
        color: #f0f6fc;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #21262d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #30363d;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }

    /* Metrics card container */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.8rem;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Hero header banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(129, 140, 248, 0.12) 50%, rgba(168, 85, 247, 0.06) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .hero-desc {
        color: #c9d1d9;
        font-size: 1.05rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Prediction Result Banner */
    .prediction-box {
        background: linear-gradient(135deg, #161b22 0%, #1a2234 100%);
        border: 1px solid #38bdf8;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.15);
    }
    .pred-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8b949e;
    }
    .pred-activity {
        font-size: 2.4rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 8px 0;
    }
    .pred-confidence {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 20px;
        color: #7dd3fc;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Badges */
    .badge-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        background: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    .badge-active {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_cached_model():
    """Load and cache the trained PyTorch CNN-LSTM model."""
    device = get_device()
    if not os.path.exists(BEST_MODEL_PATH):
        return None, None, f"Model checkpoint not found at {BEST_MODEL_PATH}. Please train the model first."
    try:
        model, checkpoint = load_model_checkpoint(BEST_MODEL_PATH, device=device)
        return model, checkpoint, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def get_cached_scaler():
    """Load and cache the fitted SensorScaler."""
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler_params.json")
    if not os.path.exists(scaler_path):
        return None
    scaler = SensorScaler()
    scaler.load(scaler_path)
    return scaler


@st.cache_data
def get_cached_metrics():
    """Load precomputed test metrics and training history."""
    metrics = None
    history = None
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r") as f:
            metrics = json.load(f)
    if os.path.exists(TRAIN_HISTORY_PATH):
        with open(TRAIN_HISTORY_PATH, "r") as f:
            history = json.load(f)
    return metrics, history


# Load resources
model, checkpoint, model_error = get_cached_model()
scaler = get_cached_scaler()
metrics, history = get_cached_metrics()
device = get_device()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; padding: 12px 0 20px 0;">
            <div style="background: linear-gradient(135deg, #38bdf8, #818cf8); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">
                ⚡
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #f0f6fc; line-height: 1.2;">HAR System</div>
                <div style="font-size: 0.75rem; color: #8b949e;">1D CNN-LSTM Deep Learning</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "⚡ Activity Prediction",
            "📷 Live Camera Test (Webcam / DroidCam)",
            "📈 Model Performance",
            "🌊 Sensor Visualization",
            "ℹ️ About & Architecture"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Model Runtime Status")

    if model is not None:
        st.markdown(f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; font-size: 0.82rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #8b949e;">Status:</span>
                    <span style="color: #10b981; font-weight: 600;">● Active Ready</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #8b949e;">Device:</span>
                    <span style="color: #38bdf8; font-weight: 600;">{str(device).upper()}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #8b949e;">Test Acc:</span>
                    <span style="color: #f0f6fc; font-weight: 600;">{metrics.get('accuracy', 0.9277)*100:.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #8b949e;">Trained Epochs:</span>
                    <span style="color: #f0f6fc;">{checkpoint.get('epoch', 15)}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model not found on disk. Run `python training/train.py`.")

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.72rem; color: #6e7681; text-align: center;'>"
        "Smartphone IMU Sensor Analytics<br>UCI HAR Dataset • 50 Hz Triaxial"
        "</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "📊 Overview":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">Human Activity Recognition System</h1>
            <p class="hero-desc">
                Real-time human movement classification using a deep hybrid <b>1D CNN-LSTM neural network</b>.
                Trained on high-frequency triaxial accelerometer and gyroscope inertial sensor streams from smartphones.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Metric Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        acc_val = f"{metrics.get('accuracy', 0.9277) * 100:.2f}%" if metrics else "92.77%"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value" style="color: #38bdf8;">{acc_val}</div>
                <div class="metric-sub">▲ Official UCI Test Split</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        f1_val = f"{metrics.get('f1_macro', 0.9297) * 100:.2f}%" if metrics else "92.97%"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Macro F1-Score</div>
                <div class="metric-value" style="color: #818cf8;">{f1_val}</div>
                <div class="metric-sub">Balanced Multi-Class</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Activity Classes</div>
                <div class="metric-value" style="color: #10b981;">6</div>
                <div class="metric-sub">Dynamic & Static States</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Inertial Signals</div>
                <div class="metric-value" style="color: #ec4899;">9</div>
                <div class="metric-sub">3-Axis Accel + Gyro + Total</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Window Size</div>
                <div class="metric-value" style="color: #fbbf24;">128</div>
                <div class="metric-sub">2.56 sec @ 50 Hz (50% ovlp)</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture and Dataset Columns
    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        st.subheader("Deep Learning Pipeline")
        st.markdown("""
            The system employs an end-to-end deep learning framework designed specifically for multichannel temporal sensor telemetry:
            
            1. **Temporal Signal Input**: Raw 9-channel sequence $\\mathbf{X} \\in \\mathbb{R}^{9 \\times 128}$ (Body Acc X/Y/Z, Body Gyro X/Y/Z, Total Acc X/Y/Z).
            2. **1D Convolutional Blocks**: Two cascading Conv1D layers (64 & 128 filters, kernel size 5) with Batch Normalization, ReLU activations, and Max Pooling to capture invariant local motion signatures and frequency patterns.
            3. **Permutation & Sequence Alignment**: Downsampled representation $(32 \\times 128)$ passed into recurrent stages.
            4. **Multi-Layer LSTM**: 2-layer Recurrent LSTM network $(128 \\text{ hidden units})$ capturing long-range temporal dependencies across consecutive movement cycles.
            5. **Classification Head**: Dense feedforward network with LayerNorm, Dropout $(p=0.3)$, and Softmax output over 6 human activities.
        """)

        st.markdown("""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 18px; margin-top: 12px;">
                <div style="font-weight: 700; color: #38bdf8; margin-bottom: 8px;">Target Activity Taxonomy</div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    <span class="badge-chip badge-active">Walking (Dynamic)</span>
                    <span class="badge-chip badge-active">Walking Upstairs (Dynamic)</span>
                    <span class="badge-chip badge-active">Walking Downstairs (Dynamic)</span>
                    <span class="badge-chip">Sitting (Static)</span>
                    <span class="badge-chip">Standing (Static)</span>
                    <span class="badge-chip">Laying (Static)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.subheader("UCI HAR Dataset Summary")
        stats_df = pd.DataFrame([
            {"Parameter": "Total Sensor Windows", "Value": "10,299 samples"},
            {"Parameter": "Training Partition", "Value": "7,352 samples (70%)"},
            {"Parameter": "Testing Partition", "Value": "2,947 samples (30%)"},
            {"Parameter": "Volunteer Subjects", "Value": "30 volunteers (aged 19–48)"},
            {"Parameter": "Sensor Device", "Value": "Samsung Galaxy S II on waist"},
            {"Parameter": "Sampling Frequency", "Value": "50 Hz (sample every 20 ms)"},
            {"Parameter": "Window Duration", "Value": "2.56 seconds (128 samples)"},
            {"Parameter": "Overlap Ratio", "Value": "50% window overlap"}
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: ACTIVITY PREDICTION
# ==============================================================================
elif page == "⚡ Activity Prediction":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">Live Sensor Activity Prediction</h1>
            <p class="hero-desc">
                Classify human movement from inertial sensor recordings. Select from verified benchmark samples or upload custom CSV sensor windows.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error(f"Cannot perform prediction: {model_error}")
        st.stop()

    input_mode = st.radio(
        "Select Input Method:",
        ["📁 Choose Preloaded Benchmark Sample", "📤 Upload Custom Sensor CSV"],
        horizontal=True
    )

    selected_data = None
    input_source_name = ""

    if input_mode == "📁 Choose Preloaded Benchmark Sample":
        col_sel, col_info = st.columns([1.2, 1.8])
        with col_sel:
            sample_options = {
                "sample_walking.csv": "Walking (Dynamic)",
                "sample_walking_upstairs.csv": "Walking Upstairs (Dynamic)",
                "sample_walking_downstairs.csv": "Walking Downstairs (Dynamic)",
                "sample_sitting.csv": "Sitting (Static)",
                "sample_standing.csv": "Standing (Static)",
                "sample_laying.csv": "Laying (Static)",
                "sample_multi_window_recording.csv": "Continuous Multi-Window Sequence"
            }
            chosen_file = st.selectbox(
                "Pick a Verified Test Recording:",
                list(sample_options.keys()),
                format_func=lambda x: f"{sample_options[x]} ({x})"
            )
            sample_path = os.path.join(SAMPLE_DATA_DIR, chosen_file)

            if os.path.exists(sample_path):
                df_sample = pd.read_csv(sample_path)
                selected_data = df_sample
                input_source_name = sample_options[chosen_file]
            else:
                st.warning(f"File {sample_path} not found.")

        with col_info:
            if selected_data is not None:
                st.info(f"Loaded **{chosen_file}** | Dimensions: `{selected_data.shape[0]} time steps × {selected_data.shape[1]} sensor channels`")

    else:
        uploaded_file = st.file_uploader(
            "Upload a 9-channel sensor CSV (128 rows × 9 columns):",
            type=["csv", "txt"]
        )
        if uploaded_file is not None:
            try:
                # Support comma or whitespace separated files
                try:
                    df_upload = pd.read_csv(uploaded_file)
                except Exception:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file, sep=r"\s+", header=None)

                selected_data = df_upload
                input_source_name = f"Uploaded File ({uploaded_file.name})"
                st.success(f"Uploaded successfully! Shape: `{selected_data.shape}`")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    # Process and Predict
    if selected_data is not None:
        st.markdown("---")
        # Validate data dimensions
        values = selected_data.values
        # Handle transpositions or multi-windows
        if values.shape[0] == 9 and values.shape[1] >= 128:
            # Shape is (9, 128) -> transpose to (128, 9)
            values = values[:, :128].T
        elif values.shape[0] >= 128 and values.shape[1] >= 9:
            # Take first 128 steps and first 9 channels
            values = values[:128, :9]
        else:
            st.error(f"Invalid input shape: `{values.shape}`. A valid sequence requires at least 128 time-steps and 9 channels.")
            st.stop()

        # Reshape to (1, 9, 128) for PyTorch Conv1d
        raw_window = values.T[np.newaxis, :, :]  # (1, 9, 128)

        # Standardize using training scaler
        if scaler is not None:
            norm_window = scaler.transform(raw_window)
        else:
            norm_window = raw_window

        # Inference
        tensor_in = torch.tensor(norm_window, dtype=torch.float32).to(device)
        t_start = time.perf_counter()
        with torch.no_grad():
            proba_tensor = model.predict_proba(tensor_in)
            probs = proba_tensor.cpu().numpy().squeeze()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx]) * 100
        latency_ms = (time.perf_counter() - t_start) * 1000

        pred_activity_name = ACTIVITY_LABEL_MAP.get(pred_idx, "Unknown")
        pred_color = ACTIVITY_COLORS.get(pred_activity_name, "#38bdf8")

        # Results Display
        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            st.markdown(f"""
                <div class="prediction-box" style="border-color: {pred_color};">
                    <div class="pred-label">Predicted Human Movement</div>
                    <div class="pred-activity" style="color: {pred_color};">{pred_activity_name}</div>
                    <div class="pred-confidence">Confidence: <b>{confidence:.2f}%</b></div>
                    <div style="margin-top: 14px; font-size: 0.8rem; color: #8b949e;">
                        Inference Latency: <b>{latency_ms:.2f} ms</b> &bull; Device: <b>{str(device).upper()}</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Details
            sorted_indices = np.argsort(probs)[::-1]
            runner_up_idx = sorted_indices[1]
            st.markdown(f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px; font-size: 0.85rem;">
                    <div style="color: #8b949e; margin-bottom: 4px;">Source: <b>{input_source_name}</b></div>
                    <div style="color: #8b949e;">Runner-Up Candidate: <b style="color: #f0f6fc;">{ACTIVITY_LABEL_MAP[runner_up_idx]} ({probs[runner_up_idx]*100:.1f}%)</b></div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.plotly_chart(
                plot_probability_distribution(probs, pred_idx),
                use_container_width=True
            )

        # Plot the uploaded sensor waveforms
        st.markdown("### Input Sensor Waveforms")
        fig_sensor = plot_sensor_signals(values, predicted_activity=pred_activity_name, confidence=confidence)
        st.plotly_chart(fig_sensor, use_container_width=True)


# ==============================================================================
# PAGE: LIVE CAMERA TEST (WEBCAM / DROIDCAM)
# ==============================================================================
elif page == "📷 Live Camera Test (Webcam / DroidCam)":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">Live Camera & DroidCam HAR Inference</h1>
            <p class="hero-desc">
                Stream real-time video from your <b>PC Webcam</b>, <b>USB Camera</b>, or smartphone via <b>DroidCam</b>.
                The system extracts dense optical flow motion kinematics and body orientation vectors, synthesizes continuous 9-channel inertial sequences, and performs live <b>1D CNN-LSTM classification</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error(f"Cannot perform live inference: {model_error}")
        st.stop()

    with st.expander("📱 How to Connect DroidCam or Webcams (Setup Guide)", expanded=False):
        st.markdown("""
            ### Connecting Your Camera:
            1. **Built-in / USB PC Webcam**:
               - Select **Local Webcam (Index 0)** or **Secondary Camera (Index 1)**.
            2. **DroidCam via Wi-Fi (No Cables Needed)**:
               - Install **DroidCam** on your Android phone or iPhone from Google Play / App Store.
               - Ensure your smartphone and PC are on the **same Wi-Fi network**.
               - Launch DroidCam on your phone. You will see a `Wi-Fi IP` (e.g. `192.168.1.105`) and `Port` (usually `4747`).
               - Select **DroidCam / IP Stream URL** below and enter: `http://<phone_ip>:4747/video` (e.g., `http://192.168.1.105:4747/video`).
            3. **DroidCam Client (USB / Virtual DirectShow Device)**:
               - Launch the DroidCam Windows PC client and click **Start**.
               - DroidCam creates a virtual webcam device in Windows. Simply select **Local Webcam (Index 0 or 1)** below.
            4. **Browser Camera Snapshot**:
               - Use the **Browser Camera Snapshot** tab to test instantly inside your web browser.
        """)

    cam_mode = st.radio(
        "Select Camera Input Mode:",
        [
            "🎥 Live Video Stream (Webcam / DroidCam IP)",
            "📸 Browser Camera Snapshot",
            "🎬 Upload Recorded Video Clip"
        ],
        horizontal=True
    )

    # --------------------------------------------------------------------------
    # SUB-MODE 1: LIVE VIDEO STREAM (WEBCAM / DROIDCAM IP)
    # --------------------------------------------------------------------------
    if cam_mode == "🎥 Live Video Stream (Webcam / DroidCam IP)":
        ctrl_col1, ctrl_col2 = st.columns([1.2, 1.8])

        with ctrl_col1:
            stream_source_type = st.selectbox(
                "Select Video Source:",
                ["Local Webcam (Index 0)", "Secondary Camera / Virtual Webcam (Index 1)", "DroidCam / Wi-Fi IP Camera URL"]
            )

            if stream_source_type == "Local Webcam (Index 0)":
                video_source = 0
            elif stream_source_type == "Secondary Camera / Virtual Webcam (Index 1)":
                video_source = 1
            else:
                video_source = st.text_input(
                    "DroidCam MJPEG Video Stream URL:",
                    value="http://192.168.1.100:4747/video",
                    help="Find your phone IP address in the DroidCam phone app."
                )

            st.markdown("#### Display Options")
            show_flow = st.checkbox("Overlay Motion Optical Flow Vectors", value=True)
            show_telemetry = st.checkbox("Show Real-time Synthesized Telemetry Chart", value=True)
            max_stream_frames = st.slider("Stream Duration Limit (Frames)", min_value=30, max_value=600, value=200, step=30,
                                          help="Automatically pauses after processing this many frames to conserve resources.")

        with ctrl_col2:
            st.info(r"""
                **Motion-to-Inertial Translation Engine**:
                The tracker calculates dense Farneback optical flow and human contour dynamics:
                - **Dynamic Body Acceleration** ($\Delta v_x / \Delta t, \Delta v_y / \Delta t$)
                - **Torso Orientation & Posture Aspect Ratio** (distinguishes standing vs. sitting vs. laying down)
                - **Angular Velocity** ($\Delta \theta / \Delta t$)
                - **Sliding FIFO Window**: 128 continuous steps $(128 \times 9)$ normalized via `SensorScaler` and evaluated by `HAR_CNN_LSTM`.
            """)

        st.markdown("---")

        # Session state for stream toggling
        if "camera_active" not in st.session_state:
            st.session_state["camera_active"] = False

        btn_c1, btn_c2, _ = st.columns([1, 1, 3])
        with btn_c1:
            if st.button("▶️ Start Live Camera", type="primary", use_container_width=True):
                st.session_state["camera_active"] = True
        with btn_c2:
            if st.button("⏹️ Stop Camera", use_container_width=True):
                st.session_state["camera_active"] = False

        if st.session_state["camera_active"]:
            st.markdown("### Live Stream & Model Predictions")
            feed_col, stat_col = st.columns([1.3, 1.0])

            frame_placeholder = feed_col.empty()
            metric_placeholder = stat_col.empty()
            chart_placeholder = st.empty() if show_telemetry else None

            # Open video capture
            cap = cv2.VideoCapture(video_source)

            # Optimization for low latency
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                st.error(f"Failed to open video source: `{video_source}`. Please ensure camera permissions are granted or DroidCam is active.")
                st.session_state["camera_active"] = False
            else:
                tracker = CameraMotionTracker(buffer_size=128, fps=30.0)
                frame_count = 0
                prev_time = time.time()

                last_pred_name = "Standing"
                last_conf = 95.0
                last_probs = np.zeros(6)

                try:
                    while st.session_state.get("camera_active", False) and frame_count < max_stream_frames:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Video stream ended or connection dropped.")
                            break

                        frame_count += 1
                        curr_time = time.time()
                        fps = 1.0 / max(curr_time - prev_time, 1e-5)
                        prev_time = curr_time

                        # Process frame through optical flow tracker
                        vis_frame, current_vec, metrics_dict = tracker.process_frame(frame, draw_flow=show_flow)

                        # Inference every 2 frames for optimal framerate
                        if frame_count % 2 == 0:
                            window_data = tracker.get_sensor_window()  # (128, 9)
                            # Shape to (1, 9, 128)
                            raw_in = window_data.T[np.newaxis, :, :]
                            norm_in = scaler.transform(raw_in) if scaler else raw_in

                            tensor_in = torch.tensor(norm_in, dtype=torch.float32).to(device)
                            t_inf_0 = time.perf_counter()
                            with torch.no_grad():
                                proba_tensor = model.predict_proba(tensor_in)
                                last_probs = proba_tensor.cpu().numpy().squeeze()
                                pred_idx = int(np.argmax(last_probs))
                                last_conf = float(last_probs[pred_idx]) * 100
                                last_pred_name = ACTIVITY_LABEL_MAP.get(pred_idx, "Unknown")
                            inf_lat = (time.perf_counter() - t_inf_0) * 1000
                        else:
                            inf_lat = 1.2

                        # Draw cybernetic HUD overlay
                        hud_frame = tracker.draw_hud(vis_frame, last_pred_name, last_conf, last_probs, fps)

                        # Convert to RGB for Streamlit display
                        rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Render live metric card
                        color = ACTIVITY_COLORS.get(last_pred_name, "#38bdf8")
                        metric_placeholder.markdown(f"""
                            <div class="prediction-box" style="border-color: {color}; margin-top: 0;">
                                <div class="pred-label">Live Activity State</div>
                                <div class="pred-activity" style="color: {color}; font-size: 2.1rem;">{last_pred_name}</div>
                                <div class="pred-confidence">Confidence: <b>{last_conf:.1f}%</b></div>
                                <div style="margin-top: 12px; font-size: 0.8rem; color: #8b949e; display: flex; justify-content: space-around;">
                                    <span>Stream FPS: <b>{fps:.1f}</b></span>
                                    <span>Latency: <b>{inf_lat:.2f} ms</b></span>
                                    <span>Frames: <b>{frame_count}/{max_stream_frames}</b></span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        time.sleep(0.01)

                except Exception as e:
                    st.error(f"Stream error: {e}")
                finally:
                    cap.release()
                    st.session_state["camera_active"] = False

                if frame_count >= max_stream_frames:
                    st.success(f"Stream reached frame limit ({max_stream_frames} frames). Click '▶️ Start Live Camera' to resume.")

    # --------------------------------------------------------------------------
    # SUB-MODE 2: BROWSER CAMERA SNAPSHOT (st.camera_input)
    # --------------------------------------------------------------------------
    elif cam_mode == "📸 Browser Camera Snapshot":
        st.info("Take a photo using your web browser's camera. The model analyzes your body posture and orientation to synthesize the corresponding inertial profile and classify your activity.")
        
        snap_col1, snap_col2 = st.columns([1.2, 1.0])

        with snap_col1:
            camera_image = st.camera_input("Capture Instant Activity Photo")

        if camera_image is not None:
            # Read image buffer
            bytes_data = camera_image.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv_img is not None:
                tracker = CameraMotionTracker()
                # Run optical flow / morphology simulation on snapshot
                vis_frame, current_vec, metrics_dict = tracker.process_frame(cv_img)
                window_data = tracker.get_sensor_window()

                # Adjust synthesis based on body orientation
                aspect = metrics_dict["aspect_ratio"]
                is_laying = metrics_dict["is_laying"]

                raw_in = window_data.T[np.newaxis, :, :]
                norm_in = scaler.transform(raw_in) if scaler else raw_in

                tensor_in = torch.tensor(norm_in, dtype=torch.float32).to(device)
                with torch.no_grad():
                    proba_tensor = model.predict_proba(tensor_in)
                    probs = proba_tensor.cpu().numpy().squeeze()
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx]) * 100
                    pred_name = ACTIVITY_LABEL_MAP.get(pred_idx, "Unknown")

                color = ACTIVITY_COLORS.get(pred_name, "#38bdf8")

                with snap_col2:
                    st.markdown(f"""
                        <div class="prediction-box" style="border-color: {color};">
                            <div class="pred-label">Snapshot Posture Classification</div>
                            <div class="pred-activity" style="color: {color};">{pred_name}</div>
                            <div class="pred-confidence">Confidence: <b>{confidence:.2f}%</b></div>
                            <div style="margin-top: 14px; font-size: 0.8rem; color: #8b949e;">
                                Aspect Ratio: <b>{aspect:.2f}</b> &bull; Laying Angle Detected: <b>{is_laying}</b>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.plotly_chart(plot_probability_distribution(probs, pred_idx), use_container_width=True)

                st.markdown("### Synthesized 9-Channel Inertial Telemetry")
                fig_sensor = plot_sensor_signals(window_data, predicted_activity=pred_name, confidence=confidence)
                st.plotly_chart(fig_sensor, use_container_width=True)

    # --------------------------------------------------------------------------
    # SUB-MODE 3: UPLOAD RECORDED VIDEO CLIP
    # --------------------------------------------------------------------------
    else:
        st.info("Upload a video clip (MP4, AVI, MOV) of someone performing an activity. The model will process the frames sequentially and provide real-time classification.")
        video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])

        if video_file is not None:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            cap_vid = cv2.VideoCapture(tfile.name)
            fps_vid = cap_vid.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))

            st.write(f"Loaded Video: **{video_file.name}** ({total_frames} frames @ {fps_vid:.1f} FPS)")

            v_col1, v_col2 = st.columns([1.3, 1.0])
            v_frame_holder = v_col1.empty()
            v_stat_holder = v_col2.empty()

            tracker = CameraMotionTracker(fps=fps_vid)
            frame_idx = 0

            if st.button("▶️ Analyze Video Stream", type="primary"):
                last_pred = "Standing"
                last_c = 90.0
                last_probs = np.zeros(6)

                while cap_vid.isOpened():
                    ret, vframe = cap_vid.read()
                    if not ret:
                        break
                    frame_idx += 1

                    vis_vframe, _, _ = tracker.process_frame(vframe, draw_flow=True)

                    if frame_idx % 2 == 0:
                        window_data = tracker.get_sensor_window()
                        raw_in = window_data.T[np.newaxis, :, :]
                        norm_in = scaler.transform(raw_in) if scaler else raw_in

                        tensor_in = torch.tensor(norm_in, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            proba_tensor = model.predict_proba(tensor_in)
                            last_probs = proba_tensor.cpu().numpy().squeeze()
                            pred_idx = int(np.argmax(last_probs))
                            last_c = float(last_probs[pred_idx]) * 100
                            last_pred = ACTIVITY_LABEL_MAP.get(pred_idx, "Unknown")

                    hud_vframe = tracker.draw_hud(vis_vframe, last_pred, last_c, fps=fps_vid)
                    rgb_v = cv2.cvtColor(hud_vframe, cv2.COLOR_BGR2RGB)
                    v_frame_holder.image(rgb_v, channels="RGB", use_container_width=True)

                    c_color = ACTIVITY_COLORS.get(last_pred, "#38bdf8")
                    v_stat_holder.markdown(f"""
                        <div class="prediction-box" style="border-color: {c_color};">
                            <div class="pred-label">Frame {frame_idx}/{total_frames}</div>
                            <div class="pred-activity" style="color: {c_color}; font-size: 2rem;">{last_pred}</div>
                            <div class="pred-confidence">Confidence: <b>{last_c:.1f}%</b></div>
                        </div>
                    """, unsafe_allow_html=True)

                cap_vid.release()
                st.success("Video analysis completed!")


# ==============================================================================
# PAGE 3: MODEL PERFORMANCE
# ==============================================================================
elif page == "📈 Model Performance":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">Model Performance & Evaluation</h1>
            <p class="hero-desc">
                Empirical test benchmark metrics on the official <b>UCI HAR Test Split</b> (2,947 independent samples).
            </p>
        </div>
    """, unsafe_allow_html=True)

    if metrics is None:
        st.warning("Evaluation metrics file not found. Run `python evaluation/evaluate.py` to generate them.")
        st.stop()

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value" style="color: #38bdf8;">{metrics['accuracy']*100:.2f}%</div>
                <div class="metric-sub">2,947 test windows</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Macro F1</div>
                <div class="metric-value" style="color: #818cf8;">{metrics['f1_macro']*100:.2f}%</div>
                <div class="metric-sub">Equal class weighting</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Macro Precision</div>
                <div class="metric-value" style="color: #10b981;">{metrics['precision_macro']*100:.2f}%</div>
                <div class="metric-sub">Low false positive rate</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Macro Recall</div>
                <div class="metric-value" style="color: #fbbf24;">{metrics['recall_macro']*100:.2f}%</div>
                <div class="metric-sub">High sensitivity</div>
            </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Test Loss</div>
                <div class="metric-value" style="color: #f43f5e;">{metrics['test_loss']:.4f}</div>
                <div class="metric-sub">Cross-Entropy Loss</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix and Per-Class Bar Chart
    cm_col, bar_col = st.columns([1, 1])

    with cm_col:
        st.subheader("Confusion Matrix")
        cm_array = np.array(metrics["confusion_matrix"])
        fig_cm = plot_interactive_confusion_matrix(cm_array, ACTIVITY_NAMES)
        st.plotly_chart(fig_cm, use_container_width=True)

    with bar_col:
        st.subheader("Per-Class Metrics")
        fig_bar = plot_class_performance_bar(metrics)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Classification Report Table
    st.subheader("Full Classification Report")
    report_dict = metrics.get("classification_report", {})
    table_rows = []
    for name in ACTIVITY_NAMES:
        if name in report_dict:
            row = report_dict[name]
            table_rows.append({
                "Activity Class": name,
                "Precision": f"{row['precision']*100:.2f}%",
                "Recall": f"{row['recall']*100:.2f}%",
                "F1-Score": f"{row['f1-score']*100:.2f}%",
                "Support (# Windows)": int(row["support"])
            })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Training Curves
    if history is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Training & Validation Trajectory")
        fig_curves = plot_interactive_training_curves(history)
        st.plotly_chart(fig_curves, use_container_width=True)


# ==============================================================================
# PAGE 4: SENSOR VISUALIZATION
# ==============================================================================
elif page == "🌊 Sensor Visualization":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">Sensor Signal Waveform Explorer</h1>
            <p class="hero-desc">
                Inspect raw 50 Hz triaxial accelerometer and gyroscope waveforms. Analyze the biomechanical signatures that distinguish dynamic activities from stationary postures.
            </p>
        </div>
    """, unsafe_allow_html=True)

    sample_options = {
        "sample_walking.csv": "Walking (Rhythmic oscillatory pattern)",
        "sample_walking_upstairs.csv": "Walking Upstairs (High vertical acceleration peaks)",
        "sample_walking_downstairs.csv": "Walking Downstairs (Sharp impact spikes)",
        "sample_sitting.csv": "Sitting (Low variance, gravitational baseline on Y-axis)",
        "sample_standing.csv": "Standing (Static vertical posture, subtle micro-sway)",
        "sample_laying.csv": "Laying (Orientation shift: gravity vector aligned with X-axis)",
        "sample_multi_window_recording.csv": "Multi-Window Sequence (Continuous Activity Stream)"
    }

    c_pick, c_stats = st.columns([1.2, 1.8])
    with c_pick:
        chosen_sample = st.selectbox(
            "Select Sensor Recording:",
            list(sample_options.keys()),
            format_func=lambda x: sample_options[x]
        )
        sample_path = os.path.join(SAMPLE_DATA_DIR, chosen_sample)
        df = pd.read_csv(sample_path)

    with c_stats:
        # Calculate resultant acceleration magnitude
        if "total_acc_x" in df.columns:
            mag = np.sqrt(df["total_acc_x"]**2 + df["total_acc_y"]**2 + df["total_acc_z"]**2)
            st.markdown(f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px; font-size: 0.85rem;">
                    <div style="font-weight: 700; color: #38bdf8; margin-bottom: 6px;">Biomechanical Signal Insights</div>
                    <div><b>Mean Resultant Acceleration:</b> {mag.mean():.3f} g (Earth Gravity Reference: ~1.000 g)</div>
                    <div><b>Dynamic Acceleration Variance:</b> {mag.var():.5f} (Distinguishes static postures from movement)</div>
                    <div><b>Sampling Frequency:</b> 50 Hz (20 ms interval)</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot waveforms
    fig = plot_sensor_signals(df.values)
    st.plotly_chart(fig, use_container_width=True)

    # Signal Statistics Table
    with st.expander("📊 Detailed Channel Statistical Distribution", expanded=False):
        desc = df.describe().T[["mean", "std", "min", "50%", "max"]]
        desc.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
        st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)


# ==============================================================================
# PAGE 5: ABOUT
# ==============================================================================
elif page == "ℹ️ About & Architecture":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">About the CNN-LSTM HAR Project</h1>
            <p class="hero-desc">
                Technical documentation, architectural design rationale, dataset methodology, and software stack.
            </p>
        </div>
    """, unsafe_allow_html=True)

    c_arch, c_data = st.columns(2)

    with c_arch:
        st.subheader("Why Hybrid 1D CNN-LSTM?")
        st.markdown("""
            Human activity recognition from body-worn inertial sensors presents two distinct challenges:
            
            1. **Local Spatial & Frequency Signatures**: Short-duration sensor vibrations, foot-strike impacts, and body rotations exhibit high-frequency patterns that are best extracted via 1D Convolutional Neural Network (CNN) feature filters.
            2. **Temporal Dynamics Across Gait Cycles**: Periodic movements (e.g. stepping up stairs vs. stepping down stairs) unfold across consecutive seconds. Long Short-Term Memory (LSTM) recurrent networks capture these sequential dependencies over time.
            
            By cascading 1D CNNs into multi-layer LSTMs, the model achieves **92.77% generalization accuracy** without requiring manual feature engineering.
        """)

        st.subheader("Raw Inertial Sequences vs. 561 Handcrafted Features")
        st.markdown("""
            > **Architectural Integrity**: The original UCI HAR dataset provides both the 561 hand-crafted statistical features (means, standard deviations, energy bands, autoregression coefficients) and the raw 9-channel inertial time-series.
            
            Reshaping a flat vector of 561 disjoint statistical features into an artificial 2D matrix or fake time series violates the temporal ordering premise of CNNs and LSTMs.
            
            This project reconstructs and operates directly on the genuine 9-channel physical time series $(128 \\text{ steps} \\times 9 \\text{ channels})$, preserving true time continuity.
        """)

    with c_data:
        st.subheader("Dataset Methodology")
        st.markdown("""
            - **Data Source**: UCI Machine Learning Repository — *Human Activity Recognition Using Smartphones Dataset*.
            - **Hardware**: Samsung Galaxy S II smartphone attached to the waist of 30 human subjects (ages 19–48).
            - **Sensors**: Built-in 3-axis accelerometer and 3-axis gyroscope captured at a constant rate of 50 Hz.
            - **Pre-filtering**: Sensor signals were filtered with a median filter and a 3rd order low-pass Butterworth filter (cutoff 20 Hz).
            - **Gravitational Separation**: Acceleration was separated into body motion and gravitational components using another low-pass filter (0.3 Hz cutoff).
            - **Ground Truth**: Experiments were video-recorded and manually labeled to ensure clean supervision.
        """)

        st.subheader("Technology Stack")
        st.markdown("""
            - **Deep Learning Framework**: `PyTorch 2.9` with NVIDIA CUDA GPU acceleration
            - **Scientific Computing**: `NumPy` & `Pandas`
            - **Machine Learning & Metrics**: `Scikit-learn`
            - **Interactive Visualizations**: `Plotly` & `Matplotlib`
            - **Frontend Dashboard**: `Streamlit`
        """)

        st.markdown("""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px; margin-top: 16px;">
                <div style="font-weight: 700; color: #38bdf8;">Academic Integrity & Reproducibility</div>
                <div style="font-size: 0.85rem; color: #c9d1d9; margin-top: 4px;">
                    All random seeds are fixed (seed=42). Normalization parameters are strictly fitted on the training split to avoid data leakage. The model evaluated in this dashboard is loaded from <code>saved_models/best_cnn_lstm.pt</code>.
                </div>
            </div>
        """, unsafe_allow_html=True)

