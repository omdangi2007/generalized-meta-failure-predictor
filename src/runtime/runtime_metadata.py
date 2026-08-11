"""Immutable runtime contracts returned by the UAIRE runtime engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RuntimeMetadata:
    architecture_family: str
    model_class: str
    total_parameters: int
    trainable_parameters: int
    number_of_output_classes: Optional[int]
    classifier_head: Optional[str]
    feature_extractor: Optional[str]
    last_convolution_layer: Optional[str]
    transformer_blocks: tuple[str, ...]
    image_size: Optional[int]
    normalization_profile: Optional[str]
    inspection_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreprocessingProfile:
    name: str
    image_size: Optional[int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    input_channels: int
    interpolation: str = "bilinear"
    source: str = "registry_default"


@dataclass(frozen=True)
class HookConfiguration:
    feature_layer: Optional[str]
    gradcam_layer: Optional[str]
    activation_layer: Optional[str]
    gradient_layer: Optional[str]
    strategy: str


@dataclass(frozen=True)
class SupportedCapabilities:
    architecture_supported: bool
    supported_explanations: tuple[str, ...]
    supported_ood_methods: tuple[str, ...]
    supported_signal_extractors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeContext:
    """Single immutable contract for runtime consumers of a PyTorch model."""

    metadata: RuntimeMetadata
    hooks: HookConfiguration
    preprocessing: PreprocessingProfile
    capabilities: SupportedCapabilities
    model: object = field(repr=False, compare=False)
