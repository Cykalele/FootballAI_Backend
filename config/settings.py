import os
import numpy as np

# =========================
# Project paths (I/O)
# =========================
# Base directory where computed metrics and intermediate artifacts are written.
OUTPUT_FOLDER = "output/metrics/"

# Output path for the final annotated video containing overlays and visual analytics.
ANNOTATED_VIDEO_PATH = "C:/Users/hause/OneDrive/01_Studium/02_Master/05_Masterarbeit/02_FootballAI/output/annotated_game.mp4"


# =========================
# Model resources
# =========================
# Base directory that stores all model files used throughout the pipeline.
BASE_PATH = "models/"

# Object detector (YOLO) fine-tuned for the football domain.
MODEL_PATH = BASE_PATH + "YOLO12s_finetuned.pt"

# Pose or field keypoint model leveraged for pitch geometry estimation and homography.
POSE_MODEL_PATH = BASE_PATH + "Pose_Model.pt"

# Convolutional autoencoder (CAE) if required for denoising or feature extraction.
CAE_PATH = BASE_PATH + "cae_model.h5"

# Re-identification model used by the tracker (e.g., BoT-SORT/DeepSORT backends).
# Alternative weights remain commented for quick switching during experiments.

REID_MODEL_PATH = BASE_PATH + "clip_market1501.pt"
# REID_MODEL_PATH = BASE_PATH + "osnet_x0_25_msmt17.pt"
# REID_MODEL_PATH = BASE_PATH + "MSMT17_clipreid.pth"


# =========================
# API configuration
# =========================
# Versioned API prefix used by the FastAPI router for all session-scoped endpoints.
API_BASE_PATH = "/api/v1/sessions"


# =========================
# Runtime / performance
# =========================
# Number of worker threads to parallelize CPU-bound parts of tracking and preprocessing.
THREAD_WORKERS = 8

# Detection confidence threshold for object detections (range: 0.0–1.0).
# Higher values reduce false positives at the potential cost of missed detections.
OBJECT_DETECTION_THRESHOLD = 0.6

# =========================
# Session management
# =========================
# Root directory where per-session artifacts are stored (logs, flags, metrics, exports).
SESSION_ROOT = "./sessions"
