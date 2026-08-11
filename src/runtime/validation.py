"""Descriptive runtime validation before consumers attach hooks or preprocess input."""

from __future__ import annotations

import math

from .runtime_metadata import RuntimeContext


class RuntimeValidationError(ValueError):
    pass


def validate_runtime_context(context: RuntimeContext, *, allow_unknown: bool = False) -> None:
    metadata, hooks, profile = context.metadata, context.hooks, context.preprocessing
    if metadata.architecture_family == "UnknownModel" and not allow_unknown:
        raise RuntimeValidationError("Unsupported model topology. Use allow_unknown=True only for generic runtime handling.")
    if metadata.classifier_head is None or metadata.number_of_output_classes is None:
        raise RuntimeValidationError("Missing classifier head: unable to determine model output classes.")
    if metadata.feature_extractor is None:
        raise RuntimeValidationError("Missing feature extractor: no usable internal feature module was discovered.")
    if hooks.feature_layer is None or hooks.activation_layer is None or hooks.gradient_layer is None:
        raise RuntimeValidationError("Missing hook configuration: unable to select a feature-producing hook layer.")
    if profile.name == "Custom" and profile.source != "caller_supplied":
        raise RuntimeValidationError("Invalid preprocessing profile: unknown models require a caller-supplied Custom profile.")
    if profile.input_channels < 1 or len(profile.mean) != profile.input_channels or len(profile.std) != profile.input_channels:
        raise RuntimeValidationError("Invalid preprocessing profile: mean/std must match input channel count.")
    if profile.image_size is not None and profile.image_size < 1:
        raise RuntimeValidationError("Invalid preprocessing profile: image_size must be positive when provided.")
    if not all(math.isfinite(value) for value in (*profile.mean, *profile.std)):
        raise RuntimeValidationError("Invalid preprocessing profile: mean/std values must be finite.")
