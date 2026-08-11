"""Build registry-backed models from inferred or checkpoint-declared metadata."""

from __future__ import annotations

import torch.nn as nn

from .registry import get_registry_entry


def build_model(architecture: str, num_classes: int, model_variant: str | None = None) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least two.")
    return get_registry_entry(architecture).constructor(model_variant, num_classes)
