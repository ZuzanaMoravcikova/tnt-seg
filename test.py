# -------------------------------------------------------------
# Author: Zuzana Moravčíková
# Project: Deep-Learning-Based Segmentation of Tunneling Nanotubes in Volumetric Bioimage Data
# Year: 2025
# Contact: 514286@mail.muni.cz
# License: MIT
# -------------------------------------------------------------

"""Test script for CSNet3D on TNT volumes.

This script:

- Loads a trained CSNet3DLightning checkpoint.
- Builds a TNT Lightning DataModule for the TNT dataset.
- Runs testing with PyTorch Lightning.
"""
import argparse
import warnings

import pytorch_lightning as pl
import torch

from data_preparation.tnt_dataset_final import TntLightningDataModule
from model.csnet_3d_lightning import CSNet3DLightning
from utils.my_typing import int_triplet
from utils.transforms import bv_train_transform, bv_test_transform

warnings.filterwarnings("ignore", category=FutureWarning, module="lightning_fabric.utilities.cloud_io")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the test script.

    Returns
    -------
    parser: Configured argument parser with all script options.

    Notes
    -----
    The main argument groups are:
    - Checkpoints (paths and names)
    - Data configuration (paths, crop/pad sizes, masking)
    - Inference/evaluation settings (thresholds, voxel spacing)
    - System/runtime options (batch size, accelerator, devices)
    """
    p = argparse.ArgumentParser(description="Test CSNet3D model on TNT volumes")

    # CHECKPOINT --------------------------------------------------------------
    p.add_argument("--ckpt_file", type=str, default="./checkpoints_thesis/quadrant_0.ckpt", help="path to .ckpt file")

    # DATA ---------------------------------------------------------------------
    p.add_argument("--data_path", type=str, required=True, help="Path to dataset root")
    p.add_argument("--final_zyx", type=int_triplet, default=(7, 128, 128), help="Final (Z,Y,X) after crop/pad")
    p.add_argument("--padding", type=int_triplet, default=(3, 90, 90), help="Padding (Z,Y,X) around instances")
    p.add_argument("--test_quadrant", type=int, default=0, help="Use this quadrant of the data for testing")
    p.add_argument("--test_context_margin_xy", type=int, default=32, help="Context margin for full-image testing")

    # INFERENCE / EVAL ---------------------------------------------------------
    p.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold for binarization")
    p.add_argument("--voxel_spacing", type=int_triplet, default=(9, 1, 1), help="Voxel spacing (Z,Y,X)")
    p.add_argument("--num_samples_to_save", type=int, default=2, help="Number of samples to save during testing")
    p.add_argument("--save_samples_path",
                   type=str, default="./results",
                   help="Path for saved test samples")

    # SYSTEM -------------------------------------------------------------------
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--accelerator", type=str, default="gpu", help="'gpu' or 'cpu'")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--args_save_path", type=str, default="args",
                   help="Where to save parsed arguments as JSON")

    return p


def main():
    """Entry point for testing CSNet3D on TNT data.

    Steps
    -----
    1. Parse command-line arguments.
    2. Resolve checkpoint path and load model.
    3. Build transforms and data module.
    4. Run Lightning test loop.
    """
    torch.set_float32_matmul_precision("medium")

    parser = build_parser()
    args = parser.parse_args()

    run_name = args.ckpt_file.split('/')[-1].replace('.ckpt', '')

    print(f"[INFO] Loading checkpoint: {args.ckpt_file}")

    model = CSNet3DLightning.load_from_checkpoint(
        args.ckpt_file,
        classes=1,
        channels=1,
        threshold=args.threshold,
        num_samples_to_save=args.num_samples_to_save,
        run_name=run_name,
        weights_only=True,
        voxel_spacing=args.voxel_spacing,
        save_samples_path=args.save_samples_path,
        log_to_logger=False,
        strict=False
    )

    train_transform = bv_train_transform()
    test_transform = bv_test_transform()

    data_module = TntLightningDataModule(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        test_transform=test_transform,
        padding=tuple(args.padding),
        masked_part=args.test_quadrant,
        test_context_margin_xy=args.test_context_margin_xy,
    )

    data_module.setup(stage="test")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
    )
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
