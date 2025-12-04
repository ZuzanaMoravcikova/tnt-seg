# code in this file is partially adapted from repository: https://github.com/iMED-Lab/CS-Net

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Parameters
    ----------
    smooth:
        Small constant added to numerator and denominator to avoid
        division by zero.

    Notes
    -----
    Expects predictions and targets to contain probabilities and binary
    labels respectively. Both will be flattened internally.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """Weighted sum of BCE and Dice losses.

    Parameters
    ----------
    alpha: Weight applied to the BCE component.
    beta: Weight applied to the Dice component.
    gamma: Kept only for backward compatibility; unused.

    Notes
    -----
    Inputs should be probabilities in ``[0, 1]``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        total_loss = self.alpha * bce_loss + self.beta * dice_loss
        return total_loss


class WbceDiceLoss(nn.Module):
    """Weighted BCE + Dice loss as described in the CS2Net paper.

    The loss is defined as:

    ``L = α * LWCE + (1 - α) * LDice``

    where LWCE is a class-balanced BCE using dynamic weights based on the
    predicted positive volume.

    Parameters
    ----------
    alpha: Mixing factor between weighted BCE and Dice.
    epsilon: Small constant used for numerical stability.

    Notes
    -----
    Inputs should be probabilities in ``[0, 1]``. The weighting factor
    for WCE is computed dynamically for each batch.
    """
    def __init__(self, alpha=0.6, epsilon=1e-5):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.contiguous()
        targets = targets.contiguous()

        p = preds.view(-1)
        g = targets.view(-1)

        sum_p = p.sum()
        N = p.numel()
        omega = (N - sum_p) / (sum_p + self.epsilon)

        wce = - (omega * g * torch.log(p + self.epsilon) + (1 - g) * torch.log(1 - p + self.epsilon))
        L_WCE = wce.mean()

        numerator = 2 * (p * g).sum() + self.epsilon
        denominator = (p ** 2).sum() + (g ** 2).sum() + self.epsilon
        L_Dice = 1 - (numerator / denominator)

        L_total = self.alpha * L_WCE + (1 - self.alpha) * L_Dice
        return L_total
