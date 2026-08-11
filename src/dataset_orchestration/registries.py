"""Supported dataset/backbone factories. Extend these registries, not execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


DATASET_REGISTRY = {
    "cifar10": (datasets.CIFAR10, 10),
    "cifar100": (datasets.CIFAR100, 100),
    "fashionmnist": (datasets.FashionMNIST, 10),
}


class IndexedVisionDataset(Dataset):
    """Keeps an immutable source index and original image alongside model input."""

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        # Default DataLoader collation cannot safely batch PIL images. Returning
        # the untouched source pixels also gives InputQualityExtractor the image
        # before model normalization.
        return self.transform(image), int(label), index, np.asarray(image).copy()


def build_dataset(name: str, root: str | Path, split: str, download: bool, normalize: bool, image_size: int | None = None):
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {sorted(DATASET_REGISTRY)}")
    dataset_type, class_count = DATASET_REGISTRY[key]
    transform_steps = []
    if image_size is not None:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    if key == "fashionmnist":
        if image_size is None:
            transform_steps.append(transforms.Resize((32, 32)))
        transform_steps.append(transforms.Grayscale(num_output_channels=3))
    transform_steps.append(transforms.ToTensor())
    if normalize and key in {"cifar10", "cifar100"}:
        transform_steps.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    base = dataset_type(root=str(root), train=(split == "train"), download=download, transform=None)
    return IndexedVisionDataset(base, transforms.Compose(transform_steps)), class_count


def _resnet18(classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def _resnet34(classes: int):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def _mobilenetv2(classes: int):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    return model


def _efficientnetb0(classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    return model


def _vit_b_16(classes: int):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, classes)
    return model


BACKBONE_REGISTRY = {
    "resnet18": _resnet18,
    "resnet34": _resnet34,
    "mobilenetv2": _mobilenetv2,
    "efficientnetb0": _efficientnetb0,
    "vit_b_16": _vit_b_16,
}


def build_backbone(name: str, checkpoint: str | Path, class_count: int, device: str):
    key = name.lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(f"Unsupported backbone '{name}'. Supported: {sorted(BACKBONE_REGISTRY)}")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = BACKBONE_REGISTRY[key](class_count)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
