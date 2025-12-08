# -------------------------------------------------------------
# Author: Zuzana Moravčíková
# Project: Deep-Learning-Based Segmentation of Tunneling Nanotubes in Volumetric Bioimage Data
# Year: 2025
# Contact: 514286@mail.muni.cz
# License: MIT
# -------------------------------------------------------------

from typing import Tuple

import numpy as np

from medpy.metric import hd
from medpy.metric.binary import hd95 as medpy_hd95
import torch
import torch.nn as nn


class MedpyHausdorffDistance(nn.Module):
    """Hausdorff distance metric using MedPy.

    Computes mean HD and HD95 over the batch using MedPy's `hd` and `hd95`.
    Inputs are expected as (N, C, D, H, W) or (N, D, H, W) tensors with
    values in [0, 1]. Thresholding is applied internally.
    """

    def __init__(self, percentile: int = 95, threshold: float = 0.5, spacing=(1, 1, 1)):
        super().__init__()
        self.percentile = percentile  # kept for API compatibility, only 95 is used
        self.threshold = threshold
        self.spacing = spacing

    @torch.no_grad()
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        preds: targets: same shape as preds.

        Returns
        -------
        (hd_mean, hd95_mean): two scalar tensors on the same device/dtype as preds.
                              If HD is undefined for all samples (e.g. empty masks),
                              NaN is returned for both.
        """
        device = preds.device
        dtype = preds.dtype

        p = preds.detach().cpu().numpy()
        t = targets.detach().cpu().numpy()

        if p.ndim == 5:
            p_bin = (p > self.threshold).astype(np.uint8)[:, 0]
            t_bin = (t > 0.5).astype(np.uint8)[:, 0]
        elif p.ndim == 4:
            p_bin = (p > self.threshold).astype(np.uint8)
            t_bin = (t > 0.5).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected prediction ndim={p.ndim}, expected 4 or 5.")

        vals_hd = []
        vals_hd95 = []

        for pb, tb in zip(p_bin, t_bin):
            if pb.sum() == 0 or tb.sum() == 0:
                continue
            try:
                vals_hd.append(float(hd(pb, tb, voxelspacing=self.spacing)))
                vals_hd95.append(float(medpy_hd95(pb, tb, voxelspacing=self.spacing)))
            except Exception:
                continue

        if len(vals_hd) == 0:
            hd_mean = float("nan")
            hd95_mean = float("nan")
        else:
            hd_mean = float(np.mean(vals_hd))
            hd95_mean = float(np.mean(vals_hd95))

        return (
            torch.tensor(hd_mean, device=device, dtype=dtype),
            torch.tensor(hd95_mean, device=device, dtype=dtype),
        )
