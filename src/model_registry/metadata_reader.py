"""Best-effort metadata recovery; missing checkpoint fields remain Unknown."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


UNKNOWN = "Unknown"


@dataclass(frozen=True)
class CheckpointMetadata:
    architecture: str = UNKNOWN
    model_variant: str = UNKNOWN
    dataset: str = UNKNOWN
    training_image_size: int | None = None
    number_of_classes: int | None = None
    checkpoint_version: str = UNKNOWN
    preprocessing_profile: str = UNKNOWN
    author: str = UNKNOWN
    training_timestamp: str = UNKNOWN


def _lookup(payload: dict[str, Any], *names: str) -> Any:
    sources = (payload, payload.get("metadata", {}), payload.get("config", {}), payload.get("hyper_parameters", {}), payload.get("hparams", {}))
    for source in sources:
        if isinstance(source, dict):
            for name in names:
                if name in source and source[name] is not None:
                    return source[name]
    return None


def read_checkpoint_metadata(payload: dict[str, Any]) -> CheckpointMetadata:
    architecture = _lookup(payload, "architecture", "architecture_family", "arch", "backbone", "model_architecture")
    variant = _lookup(payload, "model_variant", "variant", "model_name", "model")
    dataset = _lookup(payload, "dataset", "dataset_name", "training_dataset")
    image_size = _lookup(payload, "image_size", "input_size", "training_image_size")
    classes = _lookup(payload, "num_classes", "number_of_classes", "classes")
    preprocessing = _lookup(payload, "preprocessing_profile", "normalization_profile", "preprocessing")
    return CheckpointMetadata(
        architecture=str(architecture) if architecture is not None else UNKNOWN,
        model_variant=str(variant) if variant is not None else UNKNOWN,
        dataset=str(dataset) if dataset is not None else UNKNOWN,
        training_image_size=int(image_size) if isinstance(image_size, int) else None,
        number_of_classes=int(classes) if isinstance(classes, int) else None,
        checkpoint_version=str(_lookup(payload, "checkpoint_version", "version") or UNKNOWN),
        preprocessing_profile=str(preprocessing) if isinstance(preprocessing, str) else UNKNOWN,
        author=str(_lookup(payload, "author", "created_by") or UNKNOWN),
        training_timestamp=str(_lookup(payload, "training_timestamp", "created_at", "timestamp") or UNKNOWN),
    )
