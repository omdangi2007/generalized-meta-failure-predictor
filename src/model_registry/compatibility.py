"""Architecture, state-shape, classifier, and runtime compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.runtime import RuntimeEngine, RuntimeValidationError

from .registry import RegistryEntry


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    architecture: str
    model_variant: str | None
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    classifier_mismatch: bool
    runtime_compatible: bool
    torch_version_compatible: bool
    messages: tuple[str, ...]


def validate_compatibility(model: nn.Module, state_dict: dict[str, torch.Tensor], entry: RegistryEntry, architecture: str, variant: str | None) -> CompatibilityReport:
    model_state = model.state_dict()
    missing = tuple(key for key in model_state if key not in state_dict)
    unexpected = tuple(key for key in state_dict if key not in model_state)
    mismatches = tuple(
        f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_state[key].shape)}"
        for key, value in state_dict.items() if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    )
    classifier_mismatch = any(entry.expected_classifier in item for item in mismatches)
    messages = []
    if missing:
        messages.append(f"Missing keys: {len(missing)}.")
    if unexpected:
        messages.append(f"Unexpected keys: {len(unexpected)}.")
    if mismatches:
        messages.append(f"Shape mismatches: {len(mismatches)}.")
    runtime_compatible = False
    try:
        RuntimeEngine().create_context(model)
        runtime_compatible = True
    except RuntimeValidationError as error:
        messages.append(f"RuntimeEngine compatibility failed: {error}")
    compatible = not missing and not unexpected and not mismatches and runtime_compatible and entry.runtime_compatible
    return CompatibilityReport(
        compatible, architecture, variant, missing, unexpected, mismatches, classifier_mismatch,
        runtime_compatible, entry.runtime_compatible, tuple(messages),
    )
