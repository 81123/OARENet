from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import DeepGlobeRoadDataset
from losses import BCEDiceLoss
from metrics import BinarySegmentationMeter
from networks import OARENet
from utils import ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OARENet on the DeepGlobe road extraction dataset.")
    parser.add_argument("--data-root", type=str, default=r"D:\project\pythonProject\Road_Identification\SAM2-UNet\deepglobe")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    return parser.parse_args()


def build_dataloader(dataset: DeepGlobeRoadDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    optimizer: torch.optim.Optimizer | None,
    amp_enabled: bool,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    meter = BinarySegmentationMeter()
    progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        meter.update(logits.detach(), masks.detach(), loss.detach(), batch_size=batch_size)
        stats = meter.compute()
        progress.set_postfix(loss=f"{stats['loss']:.4f}", miou=f"{stats['miou']:.4f}")

    return meter.compute()


def save_checkpoint(
    checkpoint_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_miou: float,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "best_miou": best_miou,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    torch.save(state, checkpoint_dir / name)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    requested_device = args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the current Python environment.")
    device = torch.device("cuda" if requested_device == "cuda" else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp

    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    log_root = ensure_dir(args.log_dir)
    run_name = f"OARENET_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_root / run_name))

    train_set = DeepGlobeRoadDataset(args.data_root, split="train", image_size=args.image_size, augment=True)
    val_set = DeepGlobeRoadDataset(args.data_root, split="val", image_size=args.image_size, augment=False)
    train_loader = build_dataloader(train_set, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_set, args.batch_size, args.num_workers, shuffle=False)

    model = OARENet(pretrained_backbone=not args.no_pretrained_backbone).to(device)
    criterion = BCEDiceLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    start_epoch = 1
    best_miou = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_miou", -1.0))

    save_json(
        log_root / run_name / "config.json",
        {
            **vars(args),
            "device": str(device),
            "amp_enabled": amp_enabled,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
        },
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            scaler=scaler,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
            desc=f"Train {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            scaler=None,
            optimizer=None,
            amp_enabled=amp_enabled,
            desc=f"Val {epoch}/{args.epochs}",
        )
        scheduler.step()

        for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
            writer.add_scalar(f"{split_name}/loss", metrics["loss"], epoch)
            writer.add_scalar(f"{split_name}/background_iou", metrics["background_iou"], epoch)
            writer.add_scalar(f"{split_name}/road_iou", metrics["road_iou"], epoch)
            writer.add_scalar(f"{split_name}/precision", metrics["precision"], epoch)
            writer.add_scalar(f"{split_name}/recall", metrics["recall"], epoch)
            writer.add_scalar(f"{split_name}/miou", metrics["miou"], epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            name="latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_miou=max(best_miou, val_metrics["miou"]),
            args=args,
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                name=f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_miou=max(best_miou, val_metrics["miou"]),
                args=args,
            )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                name="best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_miou=best_miou,
                args=args,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_miou={train_metrics['miou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_miou={val_metrics['miou']:.4f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
