# -------------------------------------------------------------
# Author: Zuzana Moravčíková
# Project: Deep-Learning-Based Segmentation of Tunneling Nanotubes in Volumetric Bioimage Data
# Year: 2025
# Contact: 514286@mail.muni.cz
# License: MIT
# -------------------------------------------------------------

import os
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex, Specificity
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model.csnet_3d import CSNet3D
from data_preparation.tnt_regions import ensure_roi, crop_roi_batch
from utils.losses import DiceBCELoss, WbceDiceLoss
from utils.my_metrics import MedpyHausdorffDistance


class CSNet3DLightning(pl.LightningModule):
    """Lightning wrapper for CSNet3D with metrics, losses, and schedulers.

    Parameters
    ----------
    classes: Number of output segmentation channels.
    channels: Number of input image channels.
    loss: Name of the loss function to use.
    alpha: Weighting factor for mixed losses.
    weight_decay: Weight decay applied by optimizers that support it.
    lr: Initial learning rate.
    threshold: Threshold used to binarise predictions for metrics.
    num_samples_to_save: Number of test samples to save as ``.npy`` files.
    voxel_spacing: Spacing used by Hausdorff distance metrics.
    run_name: Name added to the output results folder.
    save_samples_path: Base directory for saving test samples.
    checkpoint_folder: Name appended to ``run_name`` describing the checkpoint metric.
    log_to_logger: If True, logs metrics to Lightning's logger.
    scheduler_name: Scheduler type to use.
    """

    def __init__(self,
                 classes,
                 channels,
                 loss='wbce_dice',
                 alpha=0.6,
                 weight_decay=0.0005,
                 lr=1e-4,
                 threshold=0.5,
                 num_samples_to_save=0,
                 voxel_spacing=(1, 1, 1),
                 run_name=None,
                 save_samples_path="./results",
                 checkpoint_folder=None,
                 log_to_logger=True,
                 scheduler_name="poly_epoch",
                 ):
        super(CSNet3DLightning, self).__init__()

        self.model = CSNet3D(classes, channels)
        self.lr = lr
        self.scheduler_name = scheduler_name
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.threshold = threshold
        self.voxel_spacing = voxel_spacing
        self.num_samples_to_save = num_samples_to_save
        self.log_to_logger = log_to_logger
        self.saved_samples = 0

        if loss == 'bce':
            self.criterion = nn.BCELoss()
        elif loss == 'dice_bce':
            self.criterion = DiceBCELoss(alpha=alpha)
        elif loss == 'wbce_dice':
            self.criterion = WbceDiceLoss(alpha=alpha)
        else:
            raise ValueError(f"Unknown loss function: {loss}")

        safe_run_name = run_name or "run"
        safe_checkpoint_folder = f"_{checkpoint_folder}" if checkpoint_folder else ""
        self.samples_path = os.path.join(save_samples_path, f"{safe_run_name}{safe_checkpoint_folder}")
        os.makedirs(self.samples_path, exist_ok=True)
        self.test_outputs = []

        def _build_stage_metrics(spec_average, jacc_average) -> nn.ModuleDict:
            return nn.ModuleDict({
                "acc": BinaryAccuracy(threshold=threshold),
                "sens": BinaryRecall(threshold=threshold),
                "prec": BinaryPrecision(threshold=threshold),
                "spec": Specificity(num_classes=1, threshold=threshold, average=spec_average, task='binary'),
                "jacc": JaccardIndex(num_classes=1, threshold=threshold, average=jacc_average, task='binary'),
                "h_dist": MedpyHausdorffDistance(percentile=95, threshold=threshold, spacing=voxel_spacing),
            })

        self.metrics = nn.ModuleDict({
            "train_stage": _build_stage_metrics('micro', 'macro'),
            "val_stage": _build_stage_metrics('none', 'macro'),
            "test_stage": _build_stage_metrics('none', 'macro'),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_name == "poly_epoch":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda e: (1 - e / self.trainer.max_epochs) ** 0.9
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif self.scheduler_name == "cos_warm_res":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

    def on_fit_start(self) -> None:
        self.metrics.to(self.device)

    def on_test_start(self) -> None:
        self.metrics.to(self.device)

    def _get_total_steps(self) -> int:
        """Infer total number of training steps.

        Used when losses depend on progress through training.
        """
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None or total_steps <= 0:
            if getattr(self.trainer, "max_steps", 0) and self.trainer.max_steps > 0:
                return int(self.trainer.max_steps)
            num_batches = getattr(self.trainer, "num_training_batches", 0) or 0
            max_epochs = getattr(self.trainer, "max_epochs", 0) or 0
            return int(num_batches * max_epochs) if num_batches and max_epochs else 1
        return int(total_steps)

    def _call_criterion(self, preds, targets, stage: str) -> torch.Tensor:
        """Compute the loss, passing schedule-related arguments if supported.

        Parameters
        ----------
        preds: Model predictions.
        targets: Ground-truth segmentation.
        stage: Training stage name.

        Returns
        -------
        loss: Scalar loss tensor (or dict for complex loss functions).
        """
        kwargs = {}
        if stage == "train":
            kwargs = {"step": self.global_step, "total_steps": self._get_total_steps()}
        elif stage in ("val", "test"):
            kwargs = {"epoch": self.current_epoch, "max_epochs": self.trainer.max_epochs}
        try:
            return self.criterion(preds, targets, **kwargs)
        except TypeError:
            return self.criterion(preds, targets)

    def common_step(self,
                    batch: Tuple[torch.Tensor, torch.Tensor],
                    metrics: torch.nn.Module,
                    stage: str
                    ):
        """Shared logic across train/val/test steps.

        Applies model, ROI-cropping (if present), loss calculation, and
        metric computation.

        Parameters
        ----------
        batch: Either ``(images, targets)`` or ``(images, targets, roi)``.
        metrics: Metric collection for the given stage.
        stage: Training stage name.

        Returns
        -------
        outputs: Tuple containing images, targets, predictions, loss, metrics,
                 and extra logged values if provided by the loss.
        """
        if len(batch) == 2:
            images, targets = batch
            roi = None
        else:
            images, targets, roi = batch
        preds = self(images)

        if roi is not None:
            roi = ensure_roi(targets, roi)
            images = crop_roi_batch(images, roi)
            targets = crop_roi_batch(targets, roi)
            preds = crop_roi_batch(preds, roi)

        loss_out = self._call_criterion(preds, targets, stage=stage)
        extra_logs = {}
        if isinstance(loss_out, dict):
            loss = loss_out.get("total")
            for k in ("cs2", "hd", "w_hd", "w_cs2", "progress"):
                if k in loss_out:
                    v = loss_out[k]
                    if not torch.is_tensor(v):
                        v = torch.tensor(float(v), device=preds.device, dtype=preds.dtype)
                    extra_logs[k] = v.detach()
        else:
            loss = loss_out

        acc = metrics["acc"](preds, targets)
        sens = metrics["sens"](preds, targets)
        prec = metrics["prec"](preds, targets)
        spec = metrics["spec"](preds, targets)
        jacc = metrics["jacc"](preds, targets)
        hd, hd_95 = metrics["h_dist"](preds, targets)

        return images, targets, preds, loss, acc, sens, prec, spec, jacc, hd, hd_95, extra_logs

    def log_metrics(self, loss, acc, sens, prec, spec, jacc, hd, hd_95, split_name, on_epoch=True) -> None:
        """Log a standard set of metrics for a given split.

        Parameters
        ----------
        loss: Loss value.
        acc: Accuracy value.
        sens: Sensitivity (recall) value.
        prec: Precision value.
        spec: Specificity value.
        jacc: Jaccard index value.
        hd: Hausdorff distance value.
        hd_95: 95th percentile Hausdorff distance value.
        split_name: Name of the split, e.g. ``"train"``, ``"val"``, or ``"test"``.
        on_epoch: If ``True``, log on epoch level; otherwise log on step level.
        """
        self.log(f"{split_name} loss", loss, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(f"{split_name} acc", acc, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(f"{split_name} sensitivity", sens, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(f"{split_name} precision", prec, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(f"{split_name} specificity", spec, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(f"{split_name} jaccard", jacc, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(f"{split_name} hausdorff distance", hd, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(f"{split_name} hausdorff distance 95", hd_95, on_epoch=on_epoch, on_step=not on_epoch)

    def training_step(self, batch, batch_idx):
        """Run a single training step.

        Parameters
        ----------
        batch: Batch from the training data_preparation.
        batch_idx: Index of the batch within the epoch.

        Returns
        -------
        loss: Training loss tensor used for backpropagation.
        """
        images, targets, preds, loss, acc, sens, prec, spec, jacc, hd, hd_95, extra = self.common_step(
            batch, self.metrics["train_stage"], stage="train"
        )
        self.log_metrics(loss, acc, sens, prec, spec, jacc, hd, hd_95, "train", on_epoch=False)
        if extra:
            if "cs2" in extra:
                self.log("train cs2_loss",
                         extra["cs2"],
                         on_step=True,
                         on_epoch=False,
                         prog_bar=False)
            if "w_cs2" in extra:
                self.log("train w_cs2",
                         extra["w_cs2"],
                         on_step=True,
                         on_epoch=False,
                         prog_bar=True)
            if "progress" in extra:
                self.log("train loss_progress",
                         extra["progress"],
                         on_step=True,
                         on_epoch=False,
                         prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Run a single validation step.

        Parameters
        ----------
        batch: Batch from the validation data_preparation.
        batch_idx: Index of the batch within the epoch.

        Returns
        -------
        loss: Validation loss tensor.
        """
        images, targets, preds, loss, acc, sens, prec, spec, jacc, hd, hd_95, extra = self.common_step(batch,
                                                                                                 self.metrics[
                                                                                                     "val_stage"],
                                                                                                 stage="val")
        self.log_metrics(loss, acc, sens, prec, spec, jacc, hd, hd_95, "val", on_epoch=True)
        if extra:
            if "cs2" in extra:
                self.log("val cs2_loss",
                         extra["cs2"],
                         on_step=False,
                         on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Run a single test step with optional sample saving.

        Parameters
        ----------
        batch: Batch from the test data_preparation.
        batch_idx: Index of the batch within the epoch.

        Returns
        -------
        loss: Test loss tensor.
        """
        images, targets, preds, loss, acc, sens, prec, spec, jacc, hd, hd_95, _ = self.common_step(batch,
                                                                                             self.metrics["test_stage"],
                                                                                             stage="test")
        if self.log_to_logger:
            self.log_metrics(loss, acc, sens, prec, spec, jacc, hd, hd_95, "test", on_epoch=True)

        if self.saved_samples < self.num_samples_to_save:
            image_dir = os.path.join(self.samples_path, "image")
            target_dir = os.path.join(self.samples_path, "gt")
            pred_dir = os.path.join(self.samples_path, "prediction")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(target_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            batch_size = images.shape[0]
            num_to_save = min(self.num_samples_to_save - self.saved_samples, batch_size)

            for i in range(num_to_save):
                pred_np = preds[i, :, :, :, :].squeeze().detach().cpu().numpy()
                image_np = images[i, :, :, :, :].squeeze().detach().cpu().numpy()
                target_np = targets[i, :, :, :, :].squeeze().detach().cpu().numpy()
                sample_id = f"sample_{self.saved_samples + i}"
                np.save(os.path.join(pred_dir, f"{sample_id}.npy"), pred_np)
                np.save(os.path.join(image_dir, f"{sample_id}.npy"), image_np)
                np.save(os.path.join(target_dir, f"{sample_id}.npy"), target_np)

            self.saved_samples += num_to_save

        self.test_outputs.append({
            "loss": loss, "acc": acc, "sens": sens, "prec": prec, "spec": spec, "jacc": jacc,
            "hd": hd, "hd_95": hd_95
        })
        return loss

    def on_test_epoch_end(self) -> None:
        """Aggregate and print summary statistics over all test batches."""
        losses = [x["loss"] for x in self.test_outputs]
        losses = torch.stack(losses)
        mean_loss = losses.mean().item()

        accs = [x["acc"] for x in self.test_outputs]
        accs = torch.stack(accs)
        mean_acc = accs.mean().item()

        senss = [x["sens"] for x in self.test_outputs]
        senss = torch.stack(senss)
        mean_sens = senss.mean().item()

        precs = [x["prec"] for x in self.test_outputs]
        precs = torch.stack(precs)
        mean_prec = precs.mean().item()

        specs = [x["spec"] for x in self.test_outputs]
        specs = torch.stack(specs)
        mean_spec = specs.mean().item()

        jaccs = [x["jacc"] for x in self.test_outputs]
        jaccs = torch.stack(jaccs)
        mean_jacc = jaccs.mean().item()

        hd = [x["hd"].detach().cpu().item() for x in self.test_outputs]
        mean_hd = np.mean(hd)

        hd_95 = [x["hd_95"].detach().cpu().item() for x in self.test_outputs]
        mean_hd_95 = np.mean(hd_95)

        print("--------------------------------------------------------------")
        print("TEST RESULTS")
        print(f"    Loss                  = {mean_loss:.3f}")
        print(f"    Accuracy              = {mean_acc * 100:.1f}%")
        print(f"    Sensitivity           = {mean_sens * 100:.1f}%")
        print(f"    Precision             = {mean_prec * 100:.1f}%")
        print(f"    Specificity           = {mean_spec * 100:.1f}%")
        print(f"    Jaccard index         = {mean_jacc * 100:.1f}%")
        print(f"    Hausdorff distance    = {mean_hd:.1f} voxels")
        print(f"    Hausdorff distance 95 = {mean_hd_95:.1f} voxels")
        print("---------------------------------------------------------------")
