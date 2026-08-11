"""Public zero-architecture-specific-code checkpoint-to-runtime loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

from src.runtime import PreprocessingProfile, RuntimeContext, RuntimeEngine
from src.runtime.preprocessing_profiles import PROFILES

from .architecture_factory import infer_architecture, normalize_architecture
from .checkpoint_loader import load_checkpoint
from .compatibility import CompatibilityReport, validate_compatibility
from .exceptions import CheckpointCompatibilityError, UnsupportedArchitectureError
from .metadata_reader import CheckpointMetadata, UNKNOWN, read_checkpoint_metadata
from .model_builder import build_model
from .registry import get_registry_entry


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    runtime_context: RuntimeContext
    checkpoint_metadata: CheckpointMetadata
    compatibility_report: CompatibilityReport
    capabilities: object
    warnings: tuple[str, ...]


def _infer_classes(state_dict) -> int | None:
    preferred = ("fc.weight", "classifier.weight", "classifier.1.weight", "classifier.2.weight", "heads.head.weight", "head.weight")
    for key in preferred:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])
    for key, tensor in reversed(list(state_dict.items())):
        if key.endswith("weight") and tensor.ndim == 2:
            return int(tensor.shape[0])
    return None


def load_model(
    checkpoint: str | Path,
    *,
    architecture: str | None = None,
    num_classes: int | None = None,
    preprocessing_profile: PreprocessingProfile | str | None = None,
    allow_unsafe_full_model: bool = False,
    strict: bool = True,
) -> LoadedModel:
    """Load a supported checkpoint and return model, runtime context, and audit data."""
    payload = load_checkpoint(checkpoint, allow_unsafe_full_model=allow_unsafe_full_model)
    metadata = read_checkpoint_metadata(payload.raw_metadata)
    if payload.serialized_model is not None:
        model = payload.serialized_model
        context = RuntimeEngine().create_context(model, preprocessing_profile=preprocessing_profile, allow_unknown=True)
        inferred_architecture = context.metadata.architecture_family
        entry = get_registry_entry(inferred_architecture)
        report = CompatibilityReport(True, inferred_architecture, None, (), (), (), False, True, entry.runtime_compatible, ("Trusted serialized model loaded.",))
        return LoadedModel(model, context, metadata, report, context.capabilities, payload.warnings)
    assert payload.state_dict is not None
    inferred_family, inferred_variant = infer_architecture(payload.state_dict)
    requested_family = normalize_architecture(architecture) if architecture else None
    checkpoint_family = normalize_architecture(metadata.architecture) if metadata.architecture != UNKNOWN else None
    if requested_family and requested_family != inferred_family:
        raise CheckpointCompatibilityError(f"Requested architecture {requested_family} conflicts with checkpoint topology {inferred_family}.")
    if checkpoint_family and checkpoint_family != inferred_family:
        raise CheckpointCompatibilityError(f"Checkpoint metadata architecture {checkpoint_family} conflicts with checkpoint topology {inferred_family}.")
    selected_classes = num_classes or metadata.number_of_classes or _infer_classes(payload.state_dict)
    if selected_classes is None:
        raise CheckpointCompatibilityError("Could not determine number of classes; pass num_classes explicitly.")
    variant = metadata.model_variant if metadata.model_variant != UNKNOWN else inferred_variant
    # Variant values stored as arbitrary display names are not reliable factories.
    if variant == UNKNOWN or not str(variant).replace("_", "").isalnum():
        variant = inferred_variant
    entry = get_registry_entry(inferred_family)
    try:
        model = build_model(inferred_family, selected_classes, variant)
    except (AttributeError, TypeError) as error:
        raise UnsupportedArchitectureError(f"Unsupported model variant '{variant}' for {inferred_family}.") from error
    report = validate_compatibility(model, payload.state_dict, entry, inferred_family, variant)
    if strict and not report.compatible:
        details = " ".join(report.messages) or "state_dict validation failed."
        raise CheckpointCompatibilityError(f"Checkpoint is incompatible with {inferred_family}: {details}")
    compatible_state = {key: value for key, value in payload.state_dict.items() if key in model.state_dict() and tuple(value.shape) == tuple(model.state_dict()[key].shape)}
    model.load_state_dict(compatible_state, strict=False)
    profile = preprocessing_profile
    if profile is None and metadata.preprocessing_profile != UNKNOWN:
        profile = metadata.preprocessing_profile if metadata.preprocessing_profile.lower() in PROFILES else None
    context = RuntimeEngine().create_context(model, preprocessing_profile=profile, image_size=metadata.training_image_size)
    warnings = payload.warnings + report.messages
    return LoadedModel(model, context, metadata, report, context.capabilities, warnings)
