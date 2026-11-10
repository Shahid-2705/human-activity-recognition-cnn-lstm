"""
Camera and DroidCam live streaming module for Human Activity Recognition.
Extracts optical flow kinematics and body pose dynamics from video frames,
synthesizes 9-channel inertial sensor sequences, and renders live HUD overlays.
"""

import time
from collections import deque
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

from utils.config import (
    ACTIVITY_LABEL_MAP,
    ACTIVITY_NAMES,
    ACTIVITY_COLORS,
    INERTIAL_SIGNAL_NAMES
)


# Gravity vectors (g) measured from the UCI HAR training data (waist-mounted phone).
# The vertical body axis is X: upright puts ~+1g on X, laying rotates it onto Y/Z.
UPRIGHT_GRAVITY = (0.99, -0.27, 0.14)
LAYING_GRAVITY = (-0.18, 0.74, 0.66)


def _rest_vector(gravity) -> np.ndarray:
    """Resting 9-channel reading: zero body motion + gravity on total_acc."""
    return np.array([0, 0, 0, 0, 0, 0, *gravity], dtype=np.float32)


class CameraMotionTracker:
    """
    Real-time vision-to-inertial kinematics tracker.
    Translates camera video motion (from Webcams or DroidCam streams) into
    9-channel triaxial accelerometer and gyroscope time-series signals.
    """

    def __init__(self, buffer_size: int = 128, fps: float = 30.0):
        self.buffer_size = buffer_size
        self.fps = fps
        self.dt = 1.0 / max(fps, 10.0)

        # FIFO buffer of 9-channel sensor readings: (buffer_size, 9)
        self.buffer = deque(maxlen=buffer_size)
        
        # Pre-fill buffer with initial resting state (standing baseline)
        for _ in range(buffer_size):
            self.buffer.append(_rest_vector(UPRIGHT_GRAVITY))

        # Optical flow and motion state
        self.prev_gray = None
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_angle = 90.0
        self.prev_scale = 1.0

        # Background subtractor for robust human silhouette detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=25, detectShadows=False
        )

        # Posture state with hysteresis: a person is only considered "laying"
        # after a wide, horizontal, frame-spanning silhouette persists for
        # ~1 second, and is only released after it has clearly disappeared.
        self.lay_counter = 0
        self.last_lay_candidate = False
        self.lay_on_frames = max(int(self.fps * 1.0), 10)
        self.is_laying_state = False

    def reset(self):
        """Reset internal buffers and history."""
        self.prev_gray = None
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_angle = 90.0
        self.prev_scale = 1.0
        self.lay_counter = 0
        self.last_lay_candidate = False
        self.is_laying_state = False
        self.buffer.clear()
        for _ in range(self.buffer_size):
            self.buffer.append(_rest_vector(UPRIGHT_GRAVITY))

    def process_frame(
        self,
        frame: np.ndarray,
        draw_flow: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process a video frame and synthesize the 9 inertial sensor channels.
        
        Returns:
            vis_frame: Annotated frame with motion HUD
            current_vector: (9,) current sensor values
            metrics: Dictionary of extracted motion indicators
        """
        h, w = frame.shape[:2]
        vis_frame = frame.copy()

        # Resize for fast real-time optical flow computation
        proc_w = 320
        proc_h = int(h * (proc_w / w))
        small = cv2.resize(frame, (proc_w, proc_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Silhouette & Body Orientation via MOG2
        # Low learning rate: a person who stops moving stays in the foreground longer
        fg_mask = self.bg_subtractor.apply(gray, learningRate=0.005)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find largest motion contour (human subject)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        contour_area = 0.0
        bbox = None
        body_angle = self.prev_angle
        scale = self.prev_scale

        # Ignore small blobs (hands, shadows, noise): the subject must cover at
        # least 4% of the processed frame to be used for posture estimation.
        min_body_area = 0.04 * proc_w * proc_h
        silhouette_wide = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest)
            if contour_area > min_body_area:
                largest_contour = largest
                bbox = cv2.boundingRect(largest)
                # Compute orientation from moments
                moments = cv2.moments(largest)
                if moments["m00"] > 0:
                    mu20 = moments["mu20"] / moments["m00"]
                    mu02 = moments["mu02"] / moments["m00"]
                    mu11 = moments["mu11"] / moments["m00"]
                    body_angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) * (180.0 / np.pi)

                # Aspect ratio (height / width): distinguishes standing vs sitting vs laying
                bw, bh = bbox[2], bbox[3]
                scale = float(bh) / max(float(bw), 1.0)
                # Laying requires a silhouette that spans most of the frame width
                silhouette_wide = bw > 0.6 * proc_w

        # Optical Flow for Velocity & Acceleration
        vx, vy = 0.0, 0.0
        angular_vel = 0.0
        motion_energy = 0.0

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Mask flow to motion region
            if largest_contour is not None:
                mask_bool = fg_mask > 0
                if np.any(mask_bool):
                    flow_x = flow[..., 0][mask_bool]
                    flow_y = flow[..., 1][mask_bool]
                    vx = float(np.median(flow_x)) if len(flow_x) > 0 else 0.0
                    vy = float(np.median(flow_y)) if len(flow_y) > 0 else 0.0
                    motion_energy = float(np.mean(np.sqrt(flow_x**2 + flow_y**2)))
            else:
                flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_energy = float(np.mean(flow_mag))
                vx = float(np.mean(flow[..., 0]))
                vy = float(np.mean(flow[..., 1]))

            # Draw sparse motion vectors
            if draw_flow and motion_energy > 0.3:
                step = 24
                scale_x = w / proc_w
                scale_y = h / proc_h
                for y in range(0, proc_h, step):
                    for x in range(0, proc_w, step):
                        fx, fy = flow[y, x]
                        if fx*fx + fy*fy > 1.0:
                            pt1 = (int(x * scale_x), int(y * scale_y))
                            pt2 = (int((x + fx * 2) * scale_x), int((y + fy * 2) * scale_y))
                            cv2.arrowedLine(vis_frame, pt1, pt2, (0, 230, 255), 1, tipLength=0.3)

        self.prev_gray = gray

        # Compute Accelerations (derivative of velocity scaled to g's)
        # Normalizing factor maps pixel motion to standard human g acceleration ranges
        scale_factor = 0.04
        # Sensor X is the vertical body axis, so vertical image motion drives it
        acc_x = (vy - self.prev_vy) * scale_factor
        acc_y = (vx - self.prev_vx) * scale_factor
        acc_z = (scale - self.prev_scale) * 0.08  # Depth/approach velocity derivative

        # Gyroscope: angular velocity from body angle rotation rate
        d_angle = (body_angle - self.prev_angle) * (np.pi / 180.0) / self.dt
        gyro_z = np.clip(d_angle * 0.2, -5.0, 5.0)
        gyro_x = np.clip(acc_y * 1.5, -4.0, 4.0)
        gyro_y = np.clip(acc_x * 1.5, -4.0, 4.0)

        # Gravitational vector based on posture orientation
        # When laying down: gravity shifts into X or Z plane
        # When upright: gravity lies on the vertical X axis (+0.99g)
        laying_candidate = (
            largest_contour is not None
            and silhouette_wide
            and (abs(body_angle) < 30.0 or abs(body_angle) > 150.0)
            and scale < 0.7
        )
        if largest_contour is not None:
            self.last_lay_candidate = laying_candidate
        # With no body-sized contour (person motionless and absorbed into the
        # background) the last posture evidence keeps driving the counter, so a
        # still person stays in their posture and a stood-up person is released.
        if self.last_lay_candidate:
            self.lay_counter = min(self.lay_counter + 1, self.lay_on_frames * 2)
        else:
            self.lay_counter = max(self.lay_counter - 2, 0)
        if not self.is_laying_state and self.lay_counter >= self.lay_on_frames:
            self.is_laying_state = True
        elif self.is_laying_state and self.lay_counter == 0:
            self.is_laying_state = False
        is_laying = self.is_laying_state

        # Fixed gravity direction per posture (no per-frame flipping from noise)
        grav_x, grav_y, grav_z = LAYING_GRAVITY if is_laying else UPRIGHT_GRAVITY

        # Total acceleration = dynamic body acceleration + gravity
        total_acc_x = acc_x + grav_x
        total_acc_y = acc_y + grav_y
        total_acc_z = acc_z + grav_z

        # Clamp to realistic smartphone sensor ranges
        acc_x = float(np.clip(acc_x, -2.5, 2.5))
        acc_y = float(np.clip(acc_y, -2.5, 2.5))
        acc_z = float(np.clip(acc_z, -2.5, 2.5))
        gyro_x = float(np.clip(gyro_x, -6.0, 6.0))
        gyro_y = float(np.clip(gyro_y, -6.0, 6.0))
        gyro_z = float(np.clip(gyro_z, -6.0, 6.0))
        total_acc_x = float(np.clip(total_acc_x, -3.5, 3.5))
        total_acc_y = float(np.clip(total_acc_y, -3.5, 3.5))
        total_acc_z = float(np.clip(total_acc_z, -3.5, 3.5))

        # Pack 9-channel sensor reading
        current_vector = np.array([
            acc_x, acc_y, acc_z,
            gyro_x, gyro_y, gyro_z,
            total_acc_x, total_acc_y, total_acc_z
        ], dtype=np.float32)

        # Append to temporal sliding buffer
        self.buffer.append(current_vector)

        # Update historical state
        self.prev_vx = vx
        self.prev_vy = vy
        self.prev_angle = body_angle
        self.prev_scale = scale

        # Draw bounding box and centroid if detected
        if bbox is not None:
            scale_x = w / proc_w
            scale_y = h / proc_h
            bx, by, bw, bh = int(bbox[0]*scale_x), int(bbox[1]*scale_y), int(bbox[2]*scale_x), int(bbox[3]*scale_y)
            cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2)
            cv2.circle(vis_frame, (bx + bw//2, by + bh//2), 4, (0, 255, 128), -1)

        metrics = {
            "motion_energy": motion_energy,
            "body_angle": body_angle,
            "aspect_ratio": scale,
            "vx": vx,
            "vy": vy,
            "is_laying": is_laying
        }

        return vis_frame, current_vector, metrics

    def get_sensor_window(self) -> np.ndarray:
        """
        Return the current 128-sample sensor sequence.
        Returns:
            np.ndarray: shape (128, 9)
        """
        return np.array(self.buffer, dtype=np.float32)

    def draw_hud(
        self,
        frame: np.ndarray,
        predicted_activity: str,
        confidence: float,
        probs: Optional[np.ndarray] = None,
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Render a modern cybernetic HUD overlay onto the video frame.
        """
        h, w = frame.shape[:2]
        hud = frame.copy()

        # Activity color mapping in BGR
        bgr_colors = {
            "Walking": (248, 189, 56),            # Cyan
            "Walking Upstairs": (241, 102, 99),    # Indigo
            "Walking Downstairs": (248, 140, 129), # Periwinkle
            "Sitting": (153, 72, 236),             # Pink
            "Standing": (247, 85, 168),            # Violet
            "Laying": (129, 185, 16)               # Emerald
        }
        accent_color = bgr_colors.get(predicted_activity, (248, 189, 56))

        # 1. Top HUD Banner
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (10, 14, 20), -1)
        cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
        cv2.line(hud, (0, 65), (w, 65), accent_color, 2)

        # Text labels
        cv2.putText(hud, "LIVE HAR INFERENCE", (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(hud, f"{predicted_activity.upper()}", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, accent_color, 2, cv2.LINE_AA)

        # Confidence Badge
        conf_text = f"CONFIDENCE: {confidence:.1f}%"
        cv2.putText(hud, conf_text, (w - 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1, cv2.LINE_AA)

        # Confidence Bar
        bar_x, bar_y, bar_w, bar_h = w - 220, 42, 200, 10
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 41, 59), -1)
        fill_w = int(bar_w * (confidence / 100.0))
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), accent_color, -1)
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (71, 85, 105), 1)

        # 2. Bottom Status Bar
        bot_overlay = hud.copy()
        cv2.rectangle(bot_overlay, (0, h - 35), (w, h), (10, 14, 20), -1)
        cv2.addWeighted(bot_overlay, 0.8, hud, 0.2, 0, hud)

        fps_text = f"FPS: {fps:.1f} | Window: 128x9 | 1D CNN-LSTM Active"
        cv2.putText(hud, fps_text, (18, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

        return hud

