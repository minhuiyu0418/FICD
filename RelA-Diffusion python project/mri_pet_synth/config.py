# -*- coding: utf-8 -*-
from pathlib import Path

# -------------------------
# Reproducibility
# -------------------------
SEED = 42

# -------------------------
# Data paths
# -------------------------
DATA_ROOT = Path("F:/Dataset/NFLLONG/all modalities")
MASK_PATH = Path("F:/Dataset/NFLLONG/MNI_brainmask_181_217_181.nii.gz")

# -------------------------
# TorchIO transform
# -------------------------
RESCALE_MINMAX = (-1, 1)
CROP = [11, 10, 20, 17, 0, 21]
RESIZE = (160, 180, 160)

# -------------------------
# Training
# -------------------------
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_SAMPLE = 2
NUM_WORKERS = 0

N_EPOCHS = 100
WARMUP_EPOCHS = 20

ADV_WEIGHT = 1.0

# -------------------------
# Sampling
# -------------------------
NUM_INFERENCE_STEPS = 1000
MC_SAMPLES = 5  # number of seeds / MC samples

# Folder naming matches notebook (second entry includes '*PIB' in notebook)
TRACER_NAMES = ["PBR", "*PIB", "TAU"]
