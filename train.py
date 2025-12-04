"""Training entry point for 3D TNT segmentation.

This script does:
- data module creation
- model instantiation
- arguments logging to WandB
- checkpointing
- training and testing
.
Typical usage:
    python train.py --data_path /path/to/dataset --epochs 200 ...
"""

import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from data_preparation.tnt_dataset_final import TntLightningDataModule
from model.csnet_3d_lightning import CSNet3DLightning
from utils.my_typing import int_pair, int_triplet
from utils.transforms import bv_train_transform, bv_test_transform


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for training.

    The parser covers training hyperparameters, dataset configuration,
    augmentation settings, system settings, and logging options.

    Returns
    -------
    parser: Fully configured argument parser.
    """
    p = argparse.ArgumentParser(description="Train model on TNT volumes")

    # DATA ---------------------------------------------------------------------
    p.add_argument("--data_path", type=str, required=True, help="Path to dataset root")
    p.add_argument("--test_quadrant", type=int, default=0, help="Quadrant id {0..3} for testing")
    p.add_argument("--final_zyx", type=int_triplet, default=(7, 128, 128), help="Final (Z,Y,X) after crop/pad")
    p.add_argument("--padding", type=int_triplet, default=(3, 80, 80), help="Padding (Z,Y,X) around instances")
    p.add_argument("--voxel_spacing", type=int_triplet, default=(9, 1, 1), help="Voxel spacing (Z,Y,X)")
    p.add_argument("--crop_depth", choices=["all", "tight"], default="tight", help="Crop depth strategy")
    p.add_argument("--test_context_margin_xy", type=int, default=32, help="Context margin for full-image testing")

    # AUGMENTATION -------------------------------------------------------------
    p.add_argument("--augment", type=bool, default=True, help="Enable train augmentations")
    p.add_argument("--random_patches_per_image", type=int, default=75, help="Number of random train patches per image")
    p.add_argument("--random_size_z", type=int_pair, default=(5, 8), help="Random patch Z size range [min,max]")
    p.add_argument("--random_size_xy", type=int_pair, default=(80, 128), help="Random patch XY size range [min,max]")
    p.add_argument("--min_positive_voxels", type=int, default=300, help="Min positives in a random positive patch")

    # MODEL --------------------------------------------------------------------
    p.add_argument("--model_norm", type=str, choices=["bn", "gn", "in"], default="bn", help="Norm type inside model")
    p.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to a pretrained PL checkpoint (.ckpt)")

    # OPTIMIZATION -------------------------------------------------------------
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--loss_function", type=str, default="wbce_dice", choices=["dice_bce", "bce", "wbce_dice"])
    p.add_argument("--alpha", type=float, default=0.6, help="Alpha for combined wbce_dice losses")
    p.add_argument("--scheduler_name", type=str, default="poly_epoch", choices=["poly_epoch", "cos_warm_res"])

    # SYSTEM / LOGGING ---------------------------------------------------------
    p.add_argument("--accelerator", type=str, default="gpu", help="'gpu' or 'cpu'")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--logging", type=int, default=1, help="Enable logging (1=W&B/CSV, 0=off)")
    p.add_argument("--project_name", type=str, default="DP_fully_sup_final_dataset")
    p.add_argument("--ckpt_path", type=str, default="checkpoint_TNTs_cross_val")
    p.add_argument("--args_save_path", type=str, default="args")
    p.add_argument("--num_samples_to_save", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed")

    return p


def main():
    """Entry point for model training.

    Steps
    -----
    1. Parse arguments and initialize seeds.
    2. Configure logging, WandB or CSV.
    3. Create transforms and DataModule.
    4. Instantiate the CSNet3DLightning model.
    5. Configure checkpointing and callbacks.
    6. Run training and test evaluation.
    7. Log runtime statistics and GPU memory.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.logging = bool(args.logging)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    # ----------------------------- LOGGING ------------------------------
    use_wandb = False
    if args.logging:
        if os.getenv("DISABLE_WANDB") != "1":
            try:
                import wandb
                use_wandb = True
            except ImportError:
                print("[INFO] wandb not installed. Falling back to CSVLogger.")
                use_wandb = False
        else:
            print("[INFO] DISABLE_WANDB=1 → WandB disabled.")
    else:
        print("[INFO] Logging disabled (args.logging = 0).")

    # If a previous wandb run exists, finish it safely
    if use_wandb:
        try:
            if wandb.run is not None:
                wandb.finish()
        except:
            pass

    config_to_log = {
        'data_path': args.data_path,
        'model_file': "csnet_3d_lightning_backup.py",
        'model_norm': args.model_norm,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'ckpt_path': args.ckpt_path,
        'accelerator': args.accelerator,
        'devices': args.devices,
        'padding': args.padding,
        'augment': args.augment,
        'dataset': 'tnt_dataset_final.py',
        'loss_function': args.loss_function,
        'alpha': args.alpha,
        'random_patches_per_image': args.random_patches_per_image,
        'random_size_z': tuple(args.random_size_z),
        'random_size_xy': tuple(args.random_size_xy),
        'standardization': False,
        'test_quadrant': args.test_quadrant,
        'final_zyx': tuple(args.final_zyx),
        'voxel_spacing': tuple(args.voxel_spacing),
        'min_positive_voxels': args.min_positive_voxels,
        'pretrained_ckpt': args.pretrained_ckpt,
        'crop_depth': args.crop_depth,
        'test_context_margin_xy': args.test_context_margin_xy,
        'scheduler_name': args.scheduler_name,
    }

    os.makedirs(args.ckpt_path, exist_ok=True)
    torch.autograd.set_detect_anomaly(True)

    # --------------------------- TRANSFORMS -----------------------------
    train_transform = bv_train_transform() if args.augment else None
    test_transform = bv_test_transform()

    # ---------------------------- DATAMODULE ----------------------------
    data_module = TntLightningDataModule(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        train_transform=train_transform,
        test_transform=test_transform,
        padding=tuple(args.padding),
        random_patches_per_image=args.random_patches_per_image,
        random_size_z=tuple(args.random_size_z),
        random_size_xy=tuple(args.random_size_xy),
        masked_part=args.test_quadrant,
        final_zyx=tuple(args.final_zyx),
        min_positive_voxels=args.min_positive_voxels,
        crop_depth=args.crop_depth,
        test_context_margin_xy=args.test_context_margin_xy,
    )
    data_module.setup()

    # ------------------------------- MODEL ------------------------------
    model = CSNet3DLightning(
        classes=1,
        channels=1,
        lr=args.lr,
        loss=args.loss_function,
        alpha=args.alpha,
        voxel_spacing=tuple(args.voxel_spacing),
        weight_decay=args.weight_decay,
        num_samples_to_save=args.num_samples_to_save,
        scheduler_name=args.scheduler_name,
    )

    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_ckpt}")

    # ------------------------------ LOGGER ------------------------------
    logger = None
    run_name = "local_run"

    if args.logging:
        if use_wandb:
            try:
                from pytorch_lightning.loggers import WandbLogger
                logger = WandbLogger(project=args.project_name, log_model=True)

                # update config safely
                try:
                    logger.experiment.config.update(config_to_log, allow_val_change=True)
                except Exception:
                    pass

                run_name = (
                        logger.experiment.name
                        or logger.experiment.id
                        or "wandb_run"
                )
            except Exception as e:
                print(f"[WARN] WandB unavailable ({e}). Falling back to CSVLogger.")
                use_wandb = False

        if not use_wandb:
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger(save_dir="logs", name=args.project_name)

            try:
                logger.log_hyperparams(config_to_log)
            except Exception:
                pass

            run_name = f"csv-{logger.version}"
    else:
        logger = False
        run_name = "no_logger"

    # --------------------------- CHECKPOINTS ----------------------------
    model.hparams.run_name = run_name
    checkpoint_filename = f'{run_name}'
    checkpoint_loss = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_path, 'val_loss'),
        filename=checkpoint_filename,
        monitor='val loss',
        mode='min',
        save_top_k=1,
    )
    checkpoint_jaccard = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_path, 'val_jaccard'),
        filename=checkpoint_filename,
        monitor='val jaccard',
        mode='max',
        save_top_k=1,
    )

    callbacks = [checkpoint_loss, checkpoint_jaccard]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # ----------------------------- TRAINER ------------------------------
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_jaccard.best_model_path)

    if wandb is not None:
        if wandb.run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()
