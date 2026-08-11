"""Run artifacts and content fingerprints for reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def experiment_manifest(config, checkpoint: Path, schema: tuple[str, ...]) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(),
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "schema": list(schema),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
    }


def dataset_manifest(config, source_dataset, class_count: int) -> dict:
    return {
        "dataset": config.dataset,
        "split": config.split,
        "source_root": str(Path(config.data_dir).resolve()),
        "source_dataset_class": f"{type(source_dataset.dataset).__module__}.{type(source_dataset.dataset).__name__}",
        "sample_count_available": len(source_dataset),
        "sample_count_requested": config.max_samples,
        "number_of_classes": class_count,
        "transform": repr(source_dataset.transform),
    }
