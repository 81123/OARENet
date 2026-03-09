from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path | None
    sample_id: str


class DeepGlobeRoadDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.image_size = image_size
        self.augment = augment
        self.records = self._load_records()

        if not self.records:
            raise RuntimeError(f"No samples found for split '{self.split}' in {self.root}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        mask_tensor: torch.Tensor | None = None
        if record.mask_path is not None:
            mask = Image.open(record.mask_path).convert("RGB")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_array = np.array(mask, dtype=np.uint8)
            binary_mask = (mask_array[..., 0] > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

        if self.augment and mask_tensor is not None:
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask_tensor = torch.flip(mask_tensor, dims=[2])
            if torch.rand(1).item() < 0.5:
                image = TF.vflip(image)
                mask_tensor = torch.flip(mask_tensor, dims=[1])

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        sample: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "sample_id": record.sample_id,
        }
        if mask_tensor is not None:
            sample["mask"] = mask_tensor
        return sample

    def _load_records(self) -> list[SampleRecord]:
        split_dir = self.root / self.split
        txt_file = self.root / f"{self.split}.txt"
        records: list[SampleRecord] = []

        if txt_file.exists():
            for raw_line in txt_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                image_rel = parts[0]
                mask_rel = parts[1] if len(parts) > 1 else None
                image_path = self._resolve_path(split_dir, image_rel, kind="image")
                mask_path = self._resolve_path(split_dir, mask_rel, kind="mask") if mask_rel else None
                sample_id = image_path.stem.replace("_sat", "")
                records.append(SampleRecord(image_path=image_path, mask_path=mask_path, sample_id=sample_id))
            return records

        image_dir = split_dir / "data"
        mask_dir = split_dir / "seg"
        for image_path in sorted(image_dir.glob("*_sat.*")):
            mask_path = None
            if mask_dir.exists():
                candidate = mask_dir / image_path.name.replace("_sat.jpg", "_mask.png").replace("_sat.png", "_mask.png")
                if candidate.exists():
                    mask_path = candidate
            sample_id = image_path.stem.replace("_sat", "")
            records.append(SampleRecord(image_path=image_path, mask_path=mask_path, sample_id=sample_id))
        return records

    def _resolve_path(self, split_dir: Path, relative_path: str | None, kind: str) -> Path:
        if relative_path is None:
            raise FileNotFoundError(f"Missing {kind} path for split '{self.split}'.")

        rel = Path(relative_path)
        candidates = [
            self.root / rel,
            split_dir / rel.name,
            split_dir / "data" / rel.name,
            split_dir / "seg" / rel.name,
        ]

        stem = rel.name
        if kind == "image":
            candidates.extend(
                [
                    split_dir / "data" / stem.replace(".png", ".jpg"),
                    split_dir / "data" / stem.replace(".jpg", ".png"),
                ]
            )
        else:
            candidates.append(split_dir / "seg" / stem.replace("_sat", "_mask"))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not resolve {kind} path '{relative_path}' for split '{self.split}'.")
