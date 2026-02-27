# -*- coding: utf-8 -*-
"""Model/scheduler/optimizers construction."""

import torch
from generative.inferers import DiffusionInferer
from generative.losses import RelativisticPatchAdversarialLoss
from generative.networks.nets import DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

def build(device: torch.device):
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        num_channels=[16, 32, 64],
        attention_levels=[False, False, True],
        num_head_channels=[0, 0, 64],
        num_res_blocks=2,
        norm_num_groups=8,
        use_flash_attention=True,
        with_conditioning=True,
        cross_attention_dim=64,
        num_class_embeds=4,
    ).to(device)

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=2, num_channels=4, in_channels=1, out_channels=1).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='scaled_linear_beta', beta_start=5e-4, beta_end=1.95e-2)
    inferer = DiffusionInferer(scheduler)

    adv_loss = RelativisticPatchAdversarialLoss(discriminator)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-6)

    return model, discriminator, scheduler, inferer, adv_loss, optimizer, optimizer_d
