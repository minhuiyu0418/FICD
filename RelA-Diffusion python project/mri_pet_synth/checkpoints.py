# -*- coding: utf-8 -*-
"""Checkpoint helpers."""

from pathlib import Path
import torch

def save_checkpoint(model, save_path, epoch):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, save_path)

def load_checkpoint(model, load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('epoch', 0)
