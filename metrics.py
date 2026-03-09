from __future__ import annotations

import torch


class BinarySegmentationMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.sample_count = 0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, batch_size: int) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        self.loss_sum += loss.item() * batch_size
        self.sample_count += batch_size

        self.tp += torch.sum((preds == 1) & (targets == 1)).item()
        self.fp += torch.sum((preds == 1) & (targets == 0)).item()
        self.fn += torch.sum((preds == 0) & (targets == 1)).item()
        self.tn += torch.sum((preds == 0) & (targets == 0)).item()

    def compute(self) -> dict[str, float]:
        eps = 1e-6
        road_iou = self.tp / (self.tp + self.fp + self.fn + eps)
        background_iou = self.tn / (self.tn + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        miou = (background_iou + road_iou) * 0.5
        avg_loss = self.loss_sum / max(self.sample_count, 1)

        return {
            "loss": avg_loss,
            "background_iou": background_iou,
            "road_iou": road_iou,
            "precision": precision,
            "recall": recall,
            "miou": miou,
        }
