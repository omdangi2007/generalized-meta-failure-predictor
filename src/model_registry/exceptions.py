"""Explicit errors for safe, descriptive plug-and-play checkpoint loading."""


class ModelRegistryError(ValueError):
    """Base error for model registry operations."""


class UnsupportedCheckpointError(ModelRegistryError):
    pass


class UnsupportedArchitectureError(ModelRegistryError):
    pass


class CheckpointCompatibilityError(ModelRegistryError):
    pass


class UnsafeCheckpointError(ModelRegistryError):
    pass
