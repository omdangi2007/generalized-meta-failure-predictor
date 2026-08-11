"""Strict configuration contract for generalized meta-model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MetaModelTrainingConfig:
    dataset_path: str = "results/meta_dataset/universal_meta_dataset.csv"
    output_dir: str = "results/meta_model"
    seed: int = 42
    validation_size: float = 0.2
    test_size: float = 0.2
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    class_weight: str = "balanced_subsample"
    n_jobs: int = -1
    shap_sample_size: int = 500

    def __post_init__(self) -> None:
        if not 0 < self.validation_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("validation_size and test_size must be between zero and one.")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size + test_size must be less than one.")
        if self.n_estimators < 1 or self.min_samples_leaf < 1 or self.shap_sample_size < 1:
            raise ValueError("n_estimators, min_samples_leaf, and shap_sample_size must be positive.")

    def as_dict(self) -> dict:
        return asdict(self)


def load_training_config(path: str | Path) -> MetaModelTrainingConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Training configuration not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Training configuration must be a mapping.")
    unknown = set(raw) - set(MetaModelTrainingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unsupported configuration keys: {sorted(unknown)}")
    return MetaModelTrainingConfig(**raw)
