"""Configuration loading and validation for dataset generation runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OrchestratorConfig:
    dataset: str
    backbone: str
    checkpoint: str
    output_dir: str
    split: str = "test"
    batch_size: int = 32
    data_dir: str = "data"
    model_name: str | None = None
    device: str | None = None
    num_workers: int = 0
    seed: int = 42
    max_samples: int | None = None
    download: bool = False
    normalize: bool = True
    image_size: int | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be at least 1 when supplied.")
        if self.image_size is not None and self.image_size < 1:
            raise ValueError("image_size must be positive when supplied.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> OrchestratorConfig:
    """Load JSON or YAML configuration, rejecting unknown keys."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) if source.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a mapping.")
    allowed = set(OrchestratorConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unsupported configuration keys: {sorted(unknown)}")
    return OrchestratorConfig(**payload)


def save_config(config: OrchestratorConfig, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.as_dict(), handle, sort_keys=False)
