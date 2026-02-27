# -*- coding: utf-8 -*-
"""Data loading via TorchIO. Kept consistent with notebook logic."""

from pathlib import Path
import os
import random

import nibabel as nib
import torch
import torchio as tio

# -------------------------
# Data loading (TorchIO)
# -------------------------
DATA_ROOT = Path("F:/Dataset/NFLLONG new normalised/all modalities")
MASK_PATH = Path("F:/Dataset/NFLLONG new normalised/resized_mask_181_217_181.nii.gz")

# Load brain mask
mask_img = nib.load(str(MASK_PATH))
mask = mask_img.get_fdata()


def build_datasets(data_root=DATA_ROOT, mask_path=MASK_PATH, seed: int = 0, val_size: int = 10):
    """Return (dataset_train, dataset_sample, mask_numpy)."""
    # List subject folders
    sbj_ids = os.listdir(DATA_ROOT)

    # 1) Filter subjects with all required modalities
    subjects_with_complete_modality = []
    for sbj_id in sbj_ids:
        base_path = DATA_ROOT / sbj_id
        mod1_list = list(base_path.glob("*_T1_CS.nii"))
        mod2_list = list(base_path.glob("*_PBR_CS.nii"))
        mod3_list = list(base_path.glob("*_PIB_CS.nii"))
        mod4_list = list(base_path.glob("*_TAU_CS.nii"))
        mod5_list = list(base_path.glob("w*.nii"))  # auxiliary MRI (e.g., T2-FLAIR)

        if all([len(mod1_list) == 1, len(mod2_list) == 1, len(mod3_list) == 1, len(mod4_list) == 1, len(mod5_list) == 1]):
            subjects_with_complete_modality.append(
                tio.Subject(
                    source_1=tio.ScalarImage(mod1_list[0]),  # T1
                    source_2=tio.ScalarImage(mod5_list[0]),  # T2F
                    modality_2=tio.ScalarImage(mod2_list[0]),  # PBR
                    modality_3=tio.ScalarImage(mod3_list[0]),  # PIB
                    modality_4=tio.ScalarImage(mod4_list[0])   # TAU
                )
            )

    dataset_with_complete_modality = tio.SubjectsDataset(subjects_with_complete_modality)
    print("Complete-modality dataset size:", len(dataset_with_complete_modality), "subjects")

    # 2) Choose a held-out subset of subjects
    SEED = 0
    random.seed(SEED)
    val_size = 10
    indexes = random.sample(range(len(dataset_with_complete_modality)), min(val_size, len(dataset_with_complete_modality)))
    heldout_sbjs = {str(dataset_with_complete_modality[i]["source_1"]["stem"])[:5] for i in indexes}

    # 3) Build per-tracer samples (each subject contributes up to 3 samples: PBR/PIB/TAU)
    subjects_train, subjects_sample = [], []

    def append_subject(sbj_id, src_path, tgt_path, t2f_path, label, is_sample):
        subject = tio.Subject(
            source_1=tio.ScalarImage(src_path),
            source_2=tio.ScalarImage(t2f_path),
            target_modality=tio.ScalarImage(tgt_path),
            label=label
        )
        # Apply mask (zeros background)
        subject["source_1"]["data"] *= mask
        subject["source_2"]["data"] *= mask
        subject["target_modality"]["data"] *= mask

        if is_sample:
            subjects_sample.append(subject)
        else:
            subjects_train.append(subject)

    for sbj_id in sbj_ids:
        is_sample = sbj_id in heldout_sbjs
        base_path = DATA_ROOT / sbj_id
        try:
            t1 = list(base_path.glob("*_T1_CS.nii"))[0]
            t2f = list(base_path.glob("w*"))[0]
            for label, target_glob in enumerate(["*_PBR_CS.nii", "*_PIB_CS.nii", "*_TAU_CS.nii"]):
                target_list = list(base_path.glob(target_glob))
                if target_list:
                    append_subject(sbj_id, t1, target_list[0], t2f, label, is_sample)
        except IndexError:
            continue

    dataset_train = tio.SubjectsDataset(subjects_train)
    dataset_sample = tio.SubjectsDataset(subjects_sample)
    print("Train samples:", len(dataset_train))
    print("Sampling samples:", len(dataset_sample))

    return dataset_train, dataset_sample, mask

def build_loaders(dataset_train, dataset_sample, batch_train: int = 1, batch_sample: int = 1, num_workers: int = 0):
    """Apply transforms and create dataloaders."""
    transform = tio.Compose([
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.Crop([11, 10, 20, 17, 0, 21]),
        tio.Resize((160, 180, 160)),
    ])

    training_set = tio.SubjectsDataset(dataset_train, transform=transform)
    sampling_set = tio.SubjectsDataset(dataset_sample, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True, num_workers=0)
    sample_loader = torch.utils.data.DataLoader(sampling_set, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, sample_loader, transform
