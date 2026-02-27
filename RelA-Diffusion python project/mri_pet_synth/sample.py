# -*- coding: utf-8 -*-
"""Sampling / inference (notebook logic wrapped)."""

from pathlib import Path
import torch
from torch.cuda.amp import autocast
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from .checkpoints import load_checkpoint

def sample(
    *,
    model,
    scheduler,
    inferer,
    sample_loader,
    device,
    epoch_to_load: int,
    num_inference_steps: int,
    mc_samples: int,
    tracer_names,
):
    """Run sampling and save synthesized volumes."""

    epoch_to_load = 99
    ckpt_path = f"checkpoints/epoch{epoch_to_load}_checkpoint.pt"
    epoch_loaded = load_checkpoint(model, ckpt_path)
    model.eval()

    sample_loader = sample_loader

    tracer = ["PBR", "*PIB", "TAU"]

    scheduler.set_timesteps(num_inference_steps=1000)

    for step, batch in enumerate(sample_loader):
        if step == 0: # run one example
            bsz = len(batch['source_1']['path'])

            source_1 = batch['source_1']['data'].to(device)
            source_2 = batch['source_2']['data'].to(device)
            condition = torch.cat((source_1, source_2), dim=1)

            ground_truth = batch['target_modality']['data'].to(device)
            image_class = batch['label'].to(device)

            for seed in range(5): # repeating with different seeds for MC sampling
                SEED = seed
                torch.manual_seed(SEED)

                input_noise = torch.randn((bsz, 1, 160, 180, 160), device=device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        pred_PET = inferer.sample(
                            input_noise=input_noise,
                            diffusion_model=model,
                            label=image_class,
                            scheduler=scheduler,
                            save_intermediates=False,
                            intermediate_steps=100,
                            conditioning=condition,
                        )

                # Save one synthesized volume per subject per seed
                for i in range(bsz):
                    subject_id = str(batch['target_modality']['stem'][i])

                    preds = pred_PET[i].detach().cpu()
                    target = ground_truth[i].detach().cpu()

                    PSNR = peak_signal_noise_ratio(preds, target)
                    SSIM = structural_similarity_index_measure(preds, target)
                    print("PSNR is: ", str(PSNR.item()))
                    print("SSIM is: ", str(SSIM.item()))

                    # Save synthesized 3D PET volume
                    out_dir = Path(f"synth_{tracer[int(image_class[i].item())]}/")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_pt = out_dir / f"syn_{subject_id}_seed{SEED}.pt"
                    torch.save(preds, out_pt)

    print(f"Saved synthesized outputs to: {out_dir}")
