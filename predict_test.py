from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import DeepGlobeRoadDataset
from metrics import BinarySegmentationMeter
from networks import OARENet
from utils import ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict on the DeepGlobe test split using a trained OARENet checkpoint.")
    parser.add_argument("--data-root", type=str, default=r"D:\project\pythonProject\Road_Identification\spin4\dataset")
    parser.add_argument("--checkpoint", type=str, default=r"checkpoints\epoch_040.pth")
    parser.add_argument("--output-dir", type=str, default="predictions/epoch_040")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(42)

    requested_device = args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the current Python environment.")
    device = torch.device("cuda" if requested_device == "cuda" else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint and rebuild the model with the exact training-time args.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = argparse.Namespace(**checkpoint.get("args", {}))
    model = OARENet(pretrained_backbone=getattr(ckpt_args, "no_pretrained_backbone", False) is False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ckpt_epoch = checkpoint.get("epoch", "?")

    output_dir = ensure_dir(args.output_dir)
    predictions_dir = ensure_dir(output_dir / "masks")
    overlays_dir = ensure_dir(output_dir / "overlays")

    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"  epoch          : {ckpt_epoch}")
    print(f"  checkpoint mIoU: {checkpoint.get('best_miou', 'n/a')}")
    print(f"Output dir       : {output_dir}")
    print(f"Device           : {device} (AMP={amp_enabled})")
    print(f"Threshold        : {args.threshold}")
    print()

    test_set = DeepGlobeRoadDataset(args.data_root, split="test", image_size=args.image_size, augment=False)
    print(f"Test samples     : {len(test_set)}")

    # Mean/std must match training normalization.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    meter = BinarySegmentationMeter()
    has_ground_truth = False
    records = []

    for i in tqdm(range(len(test_set)), desc="Predict", unit="img"):
        sample = test_set[i]
        sample_id = sample["sample_id"]

        image_norm = sample["image"].unsqueeze(0).to(device, non_blocking=True)  # (1,3,H,W), normalized

        # Denormalize (operate on CPU to keep it simple).
        original_image = sample["image"] * std.squeeze(0).cpu() + mean.squeeze(0).cpu()
        original_image = (original_image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(image_norm)
        probs = torch.sigmoid(logits)[0, 0]
        pred_binary = (probs >= args.threshold).float()

        # Metrics (if ground-truth mask exists for this sample).
        if "mask" in sample:
            has_ground_truth = True
            mask_tensor = sample["mask"].unsqueeze(0).to(device, non_blocking=True)
            meter.update(
                logits.detach(), mask_tensor.detach(), loss=torch.tensor(float("nan")), batch_size=1
            )

        # Save the predicted mask as a binary PNG (255 = road).
        pred_mask_np = (pred_binary.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(pred_mask_np).save(predictions_dir / f"{sample_id}_pred.png")

        # Save an overlay (prediction in red over the original image).
        overlay = original_image.copy()
        road_pixels = pred_mask_np > 127
        overlay_rgb = Image.fromarray(overlay)
        red_layer = Image.new("RGB", overlay.shape[1::-1], (255, 0, 0))
        overlay_img = Image.composite(red_layer, overlay_rgb, Image.fromarray(pred_mask_np))
        np.copyto(overlay, np.array(overlay_img))
        # Keep road region semi-transparent red so the underlying image is still visible.
        blend = original_image.copy()
        blend[road_pixels] = (
            0.5 * original_image[road_pixels] + 0.5 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)
        Image.fromarray(blend).save(overlays_dir / f"{sample_id}_overlay.png")

        records.append(
            {
                "sample_id": sample_id,
                "image": str(Path(args.data_root) / "test" / "data" / f"{sample_id}_sat.jpg"),
                "pred_mask": str(predictions_dir / f"{sample_id}_pred.png"),
            }
        )

    summary: dict = {
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt_epoch,
        "num_samples": len(test_set),
        "image_size": args.image_size,
        "threshold": args.threshold,
        "has_ground_truth": has_ground_truth,
    }

    if has_ground_truth:
        metrics = meter.compute()
        metrics["loss"] = None  # not meaningful here
        summary["metrics"] = metrics
        print()
        print("=== Test metrics ===")
        print(f"  road_iou       : {metrics['road_iou']:.4f}")
        print(f"  background_iou : {metrics['background_iou']:.4f}")
        print(f"  mIoU           : {metrics['miou']:.4f}")
        print(f"  precision      : {metrics['precision']:.4f}")
        print(f"  recall         : {metrics['recall']:.4f}")

    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"Predicted masks saved to : {predictions_dir}")
    print(f"Overlays saved to        : {overlays_dir}")
    print(f"Manifest saved to        : {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
