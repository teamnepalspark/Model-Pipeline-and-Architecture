from __future__ import annotations

import json
from pathlib import Path

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandSpatialCropd,
)


def load_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def train_transforms(patch_size: tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["t1", "b0_distorted", "b0_corrected"], image_only=False),
            EnsureChannelFirstd(keys=["t1", "b0_distorted", "b0_corrected"]),
            RandSpatialCropd(keys=["t1", "b0_distorted", "b0_corrected"], roi_size=patch_size, random_size=False),
            RandFlipd(keys=["t1", "b0_distorted", "b0_corrected"], prob=0.5, spatial_axis=0),
            RandScaleIntensityd(keys=["t1", "b0_distorted"], factors=0.1, prob=0.3),
            RandGaussianNoised(keys=["b0_distorted"], prob=0.2, std=0.01),
            EnsureTyped(keys=["t1", "b0_distorted", "b0_corrected"]),
        ]
    )


def val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["t1", "b0_distorted", "b0_corrected"], image_only=False),
            EnsureChannelFirstd(keys=["t1", "b0_distorted", "b0_corrected"]),
            EnsureTyped(keys=["t1", "b0_distorted", "b0_corrected"]),
        ]
    )


def infer_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["t1", "b0_distorted"], image_only=False),
            EnsureChannelFirstd(keys=["t1", "b0_distorted"]),
            EnsureTyped(keys=["t1", "b0_distorted"]),
        ]
    )


def build_loader(
    manifest_path: str | Path,
    transforms: Compose,
    batch_size: int,
    num_workers: int,
    cache_rate: float,
    shuffle: bool,
) -> DataLoader:
    data = load_manifest(manifest_path)
    ds = CacheDataset(data=data, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
