# -*- coding: utf-8 -*-
"""CLI entrypoint.

Recommended:
  python -m mri_pet_synth --mode all

Also supported (direct script execution):
  python src/mri_pet_synth/main.py --mode all

Direct execution works by temporarily adding the project root's `src/` to sys.path.
"""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from mri_pet_synth import config
    from mri_pet_synth.env import setup
    from mri_pet_synth.data import build_datasets, build_loaders
    from mri_pet_synth.modeling import build as build_model
    from mri_pet_synth.train import train as train_fn
    from mri_pet_synth.sample import sample as sample_fn
else:
    from . import config

    from .env import setup
    from .data import build_datasets, build_loaders
    from .modeling import build as build_model
    from .train import train as train_fn
    from .sample import sample as sample_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "sample", "all"], default="all",
                   help="What to run: train, sample, or all.")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    p.add_argument("--mask-path", type=str, default=str(config.MASK_PATH))

    p.add_argument("--n-epochs", type=int, default=config.N_EPOCHS)
    p.add_argument("--warmup-epochs", type=int, default=config.WARMUP_EPOCHS)
    p.add_argument("--adv-weight", type=float, default=config.ADV_WEIGHT)

    p.add_argument("--epoch-to-load", type=int, default=99)
    p.add_argument("--num-inference-steps", type=int, default=config.NUM_INFERENCE_STEPS)
    p.add_argument("--mc-samples", type=int, default=config.MC_SAMPLES)

    return p.parse_args()


def main():
    args = parse_args()

    device = setup(args.seed)

    dataset_train, dataset_sample, _mask = build_datasets(
        data_root=args.data_root,
        mask_path=args.mask_path,
        seed=0,
        val_size=10,
    )
    train_loader, sample_loader, _transform = build_loaders(
        dataset_train,
        dataset_sample,
        batch_train=config.BATCH_SIZE_TRAIN,
        batch_sample=config.BATCH_SIZE_SAMPLE,
        num_workers=config.NUM_WORKERS,
    )

    model, discriminator, scheduler, inferer, adv_loss, optimizer, optimizer_d = build_model(device)

    if args.mode in ("train", "all"):
        train_fn(
            model=model,
            discriminator=discriminator,
            scheduler=scheduler,
            inferer=inferer,
            adv_loss=adv_loss,
            optimizer=optimizer,
            optimizer_d=optimizer_d,
            train_loader=train_loader,
            device=device,
            n_epochs=args.n_epochs,
            warmup_epochs=args.warmup_epochs,
            adv_weight=args.adv_weight,
        )

    if args.mode in ("sample", "all"):
        sample_fn(
            model=model,
            scheduler=scheduler,
            inferer=inferer,
            sample_loader=sample_loader,
            device=device,
            epoch_to_load=args.epoch_to_load,
            num_inference_steps=args.num_inference_steps,
            mc_samples=args.mc_samples,
            tracer_names=config.TRACER_NAMES,
        )


if __name__ == "__main__":
    main()
