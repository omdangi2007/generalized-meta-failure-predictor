"""Public plug-and-play checkpoint loading API for UAIRE."""

from .compatibility import CompatibilityReport
from .exceptions import (CheckpointCompatibilityError, ModelRegistryError, UnsafeCheckpointError,
                         UnsupportedArchitectureError, UnsupportedCheckpointError)
from .loader import LoadedModel, load_model
from .metadata_reader import CheckpointMetadata
from .registry import MODEL_REGISTRY, RegistryEntry, register_architecture

__all__ = [
    "CheckpointCompatibilityError", "CheckpointMetadata", "CompatibilityReport", "LoadedModel",
    "MODEL_REGISTRY", "ModelRegistryError", "RegistryEntry", "UnsafeCheckpointError",
    "UnsupportedArchitectureError", "UnsupportedCheckpointError", "load_model", "register_architecture",
]
