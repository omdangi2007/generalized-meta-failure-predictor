"""Extensible family registry for supported checkpoint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RegistryEntry:
    family: str
    constructor: Callable[[str | None, int], nn.Module]
    supported_checkpoint_formats: tuple[str, ...]
    expected_classifier: str
    runtime_compatible: bool
    supported_explanations: tuple[str, ...]
    supported_preprocessing_profiles: tuple[str, ...]
    supported_ood_methods: tuple[str, ...]
    minimum_torch_version: str


def _version_at_least(required: str) -> bool:
    current = torch.__version__.split("+")[0]
    return tuple(map(int, current.split(".")[:2])) >= tuple(map(int, required.split(".")[:2]))


def _torchvision_constructor(factory_name: str) -> Callable[[str | None, int], nn.Module]:
    def construct(variant: str | None, classes: int) -> nn.Module:
        from torchvision import models
        factory = getattr(models, variant or factory_name)
        model = factory(weights=None)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, classes)
        elif hasattr(model, "classifier"):
            classifier = model.classifier
            if isinstance(classifier, nn.Sequential):
                for index in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[index], nn.Linear):
                        classifier[index] = nn.Linear(classifier[index].in_features, classes)
                        break
            elif isinstance(classifier, nn.Linear):
                model.classifier = nn.Linear(classifier.in_features, classes)
        elif hasattr(model, "heads") and hasattr(model.heads, "head"):
            model.heads.head = nn.Linear(model.heads.head.in_features, classes)
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, classes)
        else:
            raise ValueError(f"Could not configure a classifier head for {type(model).__name__}.")
        return model
    return construct


def _unknown_constructor(variant: str | None, classes: int) -> nn.Module:
    from .exceptions import UnsupportedArchitectureError
    raise UnsupportedArchitectureError(
        "UnknownModel cannot be reconstructed from a bare state_dict. "
        "Register a constructor for its architecture or load a trusted serialized model explicitly."
    )


_formats = (".pt", ".pth", ".ckpt", "state_dict", "serialized_model")
_cnn = ("gradcam", "lime", "shap")
_transformer = ("lime", "shap")
_profiles = ("ImageNet", "CIFAR10", "CIFAR100", "FashionMNIST", "Custom")
_ood = ("msp", "energy", "mahalanobis")
MODEL_REGISTRY = {
    "ResNet": RegistryEntry("ResNet", _torchvision_constructor("resnet18"), _formats, "fc", True, _cnn, _profiles, _ood, "2.0"),
    "DenseNet": RegistryEntry("DenseNet", _torchvision_constructor("densenet121"), _formats, "classifier", True, _cnn, _profiles, _ood, "2.0"),
    "EfficientNet": RegistryEntry("EfficientNet", _torchvision_constructor("efficientnet_b0"), _formats, "classifier", True, _cnn, _profiles, _ood, "2.0"),
    "MobileNet": RegistryEntry("MobileNet", _torchvision_constructor("mobilenet_v2"), _formats, "classifier", True, _cnn, _profiles, _ood, "2.0"),
    "ConvNeXt": RegistryEntry("ConvNeXt", _torchvision_constructor("convnext_tiny"), _formats, "classifier", True, _cnn, _profiles, _ood, "2.0"),
    "VisionTransformer": RegistryEntry("VisionTransformer", _torchvision_constructor("vit_b_16"), _formats, "heads.head", True, _transformer, _profiles, _ood, "2.0"),
    "SwinTransformer": RegistryEntry("SwinTransformer", _torchvision_constructor("swin_t"), _formats, "head", True, _transformer, _profiles, _ood, "2.0"),
    "UnknownModel": RegistryEntry("UnknownModel", _unknown_constructor, _formats, "Unknown", False, ("lime", "shap"), ("Custom",), ("msp", "energy"), "2.0"),
}


def get_registry_entry(family: str) -> RegistryEntry:
    from .exceptions import UnsupportedArchitectureError
    try:
        entry = MODEL_REGISTRY[family]
    except KeyError as error:
        raise UnsupportedArchitectureError(f"Unsupported architecture family '{family}'. Available: {sorted(MODEL_REGISTRY)}") from error
    if not _version_at_least(entry.minimum_torch_version):
        raise UnsupportedArchitectureError(f"{family} requires torch >= {entry.minimum_torch_version}; found {torch.__version__}.")
    return entry


def register_architecture(entry: RegistryEntry) -> None:
    """Register an additional family without changing loader logic."""
    MODEL_REGISTRY[entry.family] = entry
