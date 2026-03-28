"""
Global configuration for ISR Gating for Safer Cardiovascular AI.
Edit paths and hyperparameters here before running.
"""

import os

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "/data/MIMIC-CARDIAC-EXT/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs/")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "yikuan8/Clinical-Longformer"
MAX_LENGTH = 4096

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EPOCHS = 8
SEED = 42
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

# ── ISR Gating ────────────────────────────────────────────────────────────────
H_STAR_ISR = 0.05       # ISR-only target error ≤ 5%
H_STAR_HYBRID = 0.10    # Hybrid target error ≤ 10%
B_CLIP = 6.0            # Delta clip bound (nats)
M_PERM = 6              # Number of section permutations
PRIOR_MODE = "q25"      # Robust prior: q25 = 25th-percentile of permuted probs

# ── Section headers used in clinical text ─────────────────────────────────────
EVIDENCE_HEADERS = [
    "Laboratory Results:",
    "ECG:",
    "ECG Reports:",
    "Echo:",
    "Chest X-ray:",
    "CT:",
    "MRI:",
]

CONTEXT_HEADERS = [
    "Chief Complaint:",
    "History:",
    "Physical Exam:",
]

SPECIAL_TOKENS = [
    "[EVIDENCE]",
    "[CONTEXT]",
    *EVIDENCE_HEADERS,
]
