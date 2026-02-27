# -*- coding: utf-8 -*-
"""Training loop (notebook logic wrapped)."""

import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from .checkpoints import save_checkpoint

def train(
    *,
    model,
    discriminator,
    scheduler,
    inferer,
    adv_loss,
    optimizer,
    optimizer_d,
    train_loader,
    device,
    n_epochs: int,
    warmup_epochs: int,
    adv_weight: float,
):
    """Run training. Code is kept in the same order as the notebook."""
    scaler = GradScaler()
    total_start = time.time()
    for epoch_ in range(n_epochs):
        epoch = epoch_ + 1
        model.train()
        epoch_loss = 0
        epoch_noise_loss = 0
        epoch_x0_pred_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=160)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            source_1 = batch['source_1']['data'].to(device)
            source_2 = batch['source_2']['data'].to(device)
            images = batch['target_modality']['data'].to(device)
            image_class = batch["label"].to(device)

            condition = torch.cat((source_1,source_2),1)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)
                x0_pred = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                noise_pred = inferer(inputs=images, diffusion_model=model, label=image_class, noise=noise, timesteps=timesteps, condition=condition) 
                noised_image = scheduler.add_noise(original_samples = images, noise=noise, timesteps=timesteps)

                for n in range (len(noise_pred)):
                    _, x0_pred[n] = scheduler.step(torch.unsqueeze(noise_pred[n,:,:,:,:], 0), timesteps[n], torch.unsqueeze(noised_image[n,:,:,:,:], 0))
                    if image_class[n] == 3:
                        x0_pred[n] = x0_pred[n]*0.1
                        images[n] = images[n]*0.1

                noise_loss = F.mse_loss(noise_pred.float(), noise.float())
                x0_pred_loss = F.l1_loss(x0_pred,images)
                loss = noise_loss + x0_pred_loss

                if epoch > warmup_epochs:
                    logits_real = discriminator(images.contiguous().float())[-1]
                    logits_fake = discriminator(x0_pred.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_real, logits_fake, images, x0_pred, for_discriminator=False)
                    loss += adv_weight * generator_loss

                PSNR_mid = peak_signal_noise_ratio(x0_pred, images)
                SSIM_mid = structural_similarity_index_measure(x0_pred, images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch > warmup_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(x0_pred.contiguous().detach())[-1]
                logits_real = discriminator(images.contiguous().detach())[-1]
                discriminator_loss = adv_loss(logits_real, logits_fake, images, x0_pred, for_discriminator=True)
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += loss.item()
            epoch_noise_loss += noise_loss.item()
            epoch_x0_pred_loss += x0_pred_loss.item()
            if epoch > warmup_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()
            epoch_psnr += PSNR_mid.item()
            epoch_ssim += SSIM_mid.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1),
                                      "noise_loss": epoch_noise_loss / (step + 1), 
                                      "x0_pred_loss": epoch_x0_pred_loss / (step + 1),
                                      "gen_loss": gen_epoch_loss / (step + 1),
                                      "disc_loss": disc_epoch_loss / (step + 1),
                                      "PSNR": epoch_psnr / (step + 1),
                                      "SSIM": epoch_ssim / (step + 1)})

        save_path = 'checkpoints/epoch'+str(epoch)+'_checkpoint.pt'
        save_checkpoint(model, save_path, epoch)

        save_path_discriminator = 'discriminator_checkpoints/epoch'+str(epoch)+'_checkpoint.pt'
        save_checkpoint(discriminator, save_path_discriminator, epoch)

    total_time = time.time() - total_start
    print(f"Training completed.")
