"""Central, additive execution abstraction for architecture-aware UAIRE runtime use."""

from __future__ import annotations

import torch.nn as nn

from .hook_selector import HookSelector
from .model_inspector import ModelInspector
from .preprocessing_profiles import PreprocessingProfile, get_preprocessing_profile
from .runtime_metadata import RuntimeContext, SupportedCapabilities
from .runtime_registry import get_runtime_strategy
from .validation import validate_runtime_context


class RuntimeEngine:
    """Inspect a PyTorch model and return a complete immutable RuntimeContext."""

    def __init__(self, inspector: ModelInspector | None = None, hook_selector: HookSelector | None = None):
        self.inspector = inspector or ModelInspector()
        self.hook_selector = hook_selector or HookSelector()

    def create_context(
        self,
        model: nn.Module,
        *,
        preprocessing_profile: PreprocessingProfile | str | None = None,
        image_size: int | None = None,
        allow_unknown: bool = False,
    ) -> RuntimeContext:
        family = self.inspector.inspect(model).architecture_family
        strategy = get_runtime_strategy(family)
        profile = self._resolve_profile(preprocessing_profile, strategy.preprocessing_profile)
        metadata = self.inspector.inspect(
            model,
            normalization_profile=profile.name,
            image_size=image_size if image_size is not None else profile.image_size,
        )
        hooks = self.hook_selector.select(model, metadata)
        warnings = list(metadata.inspection_warnings)
        supported = metadata.architecture_family != "UnknownModel"
        if not supported:
            warnings.append("Unknown architecture uses generic feature discovery and requires an explicit Custom preprocessing profile.")
        context = RuntimeContext(
            metadata=metadata,
            hooks=hooks,
            preprocessing=profile,
            capabilities=SupportedCapabilities(
                architecture_supported=supported,
                supported_explanations=strategy.supported_explanations,
                supported_ood_methods=strategy.supported_ood_methods,
                supported_signal_extractors=strategy.supported_signal_extractors,
                warnings=tuple(warnings),
            ),
            model=model,
        )
        validate_runtime_context(context, allow_unknown=allow_unknown)
        return context

    @staticmethod
    def _resolve_profile(value: PreprocessingProfile | str | None, default_name: str) -> PreprocessingProfile:
        if value is None:
            return get_preprocessing_profile(default_name)
        return get_preprocessing_profile(value) if isinstance(value, str) else value

    # Short alias for applications that use a builder-style runtime boundary.
    build = create_context
