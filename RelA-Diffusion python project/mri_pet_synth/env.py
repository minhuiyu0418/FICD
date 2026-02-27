# -*- coding: utf-8 -*-
"""Environment setup: imports, determinism, device."""

import os
import time
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.losses import RelativisticPatchAdversarialLoss
from generative.networks.nets import DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

import random
import numpy as np
import torch
from monai.utils import set_determinism

def setup(seed: int):
    """Set seeds/determinism, return torch.device."""
    # Reproducibility
    SEED = 42
    set_determinism(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    return device
