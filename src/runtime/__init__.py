"""Public API for UAIRE's model-agnostic runtime abstraction layer."""

from .model_inspector import ModelInspector
from .preprocessing_profiles import custom_profile, get_preprocessing_profile
from .runtime_engine import RuntimeEngine
from .runtime_metadata import HookConfiguration, PreprocessingProfile, RuntimeContext, RuntimeMetadata, SupportedCapabilities
from .validation import RuntimeValidationError, validate_runtime_context

__all__ = [
    "HookConfiguration", "ModelInspector", "PreprocessingProfile", "RuntimeContext",
    "RuntimeEngine", "RuntimeMetadata", "RuntimeValidationError", "SupportedCapabilities",
    "custom_profile", "get_preprocessing_profile", "validate_runtime_context",
]
