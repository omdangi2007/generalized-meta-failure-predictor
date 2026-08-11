"""Dataset-neutral, reusable image preprocessing profile definitions."""

from __future__ import annotations

from .runtime_metadata import PreprocessingProfile


IMAGE_NET = PreprocessingProfile("ImageNet", 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 3)
CIFAR10 = PreprocessingProfile("CIFAR10", 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 3)
CIFAR100 = PreprocessingProfile("CIFAR100", 32, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 3)
FASHION_MNIST = PreprocessingProfile("FashionMNIST", 32, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 3)
CUSTOM = PreprocessingProfile("Custom", None, (), (), 0, source="caller_required")
PROFILES = {profile.name.lower(): profile for profile in (IMAGE_NET, CIFAR10, CIFAR100, FASHION_MNIST, CUSTOM)}


def get_preprocessing_profile(name: str) -> PreprocessingProfile:
    try:
        return PROFILES[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown preprocessing profile '{name}'. Available: {sorted(PROFILES)}") from error


def custom_profile(*, image_size: int | None, mean: tuple[float, ...], std: tuple[float, ...], input_channels: int) -> PreprocessingProfile:
    if len(mean) != len(std) or len(mean) != input_channels:
        raise ValueError("Custom profile mean, std, and input_channels must have matching lengths.")
    if image_size is not None and image_size < 1:
        raise ValueError("Custom profile image_size must be positive when provided.")
    return PreprocessingProfile("Custom", image_size, mean, std, input_channels, source="caller_supplied")
