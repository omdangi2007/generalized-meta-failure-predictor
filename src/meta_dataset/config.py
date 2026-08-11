"""Configuration for a reproducible multi-run meta-dataset build."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CombinationConfig:
    id: str
    dataset: str
    backbone: str
    checkpoint: str
    model_name: str
    image_size: int | None = None


DEFAULT_COMBINATIONS = (
    CombinationConfig("resnet18_cifar10", "cifar10", "resnet18", "models/resnet18_cifar10.pth", "resnet18_cifar10"),
    CombinationConfig("resnet34_cifar10", "cifar10", "resnet34", "models/resnet34_cifar10.pth", "resnet34_cifar10"),
    CombinationConfig("mobilenetv2_cifar10", "cifar10", "mobilenetv2", "models/mobilenetv2_cifar10.pth", "mobilenetv2_cifar10"),
    CombinationConfig("efficientnetb0_cifar10", "cifar10", "efficientnetb0", "models/efficientnetb0_cifar10.pth", "efficientnetb0_cifar10"),
    CombinationConfig("resnet18_cifar100", "cifar100", "resnet18", "models/resnet18_cifar100.pth", "resnet18_cifar100"),
    CombinationConfig("resnet18_fashionmnist", "fashionmnist", "resnet18", "models/resnet18_fashionmnist.pth", "resnet18_fashionmnist"),
    CombinationConfig("vit_b_16_cifar10", "cifar10", "vit_b_16", "models/vit_cifar10.pth", "vit_b_16_cifar10", image_size=224),
)


@dataclass(frozen=True)
class MetaDatasetConfig:
    output_dir: str = "results/meta_dataset"
    data_dir: str = "data"
    split: str = "test"
    batch_size: int = 32
    device: str | None = None
    num_workers: int = 0
    seed: int = 42
    max_samples: int | None = None
    download: bool = False
    normalize: bool = True
    combinations: tuple[CombinationConfig, ...] = field(default_factory=lambda: DEFAULT_COMBINATIONS)

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative.")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be positive when supplied.")
        if not self.combinations:
            raise ValueError("At least one combination is required.")

    def as_dict(self) -> dict:
        return asdict(self)


def load_meta_config(path: str | Path) -> MetaDatasetConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Meta dataset configuration must be a mapping.")
    combinations = raw.pop("combinations", None)
    allowed = set(MetaDatasetConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported configuration keys: {sorted(unknown)}")
    parsed = tuple(CombinationConfig(**item) for item in combinations) if combinations is not None else DEFAULT_COMBINATIONS
    return MetaDatasetConfig(combinations=parsed, **raw)
