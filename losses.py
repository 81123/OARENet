from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = torch.sum(probs * targets, dim=(1, 2, 3))
        denominator = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
        dice = dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice
