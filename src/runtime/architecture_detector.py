"""Topology-based recognition of common PyTorch vision architecture families."""

from __future__ import annotations

import torch.nn as nn


SUPPORTED_FAMILIES = ("ResNet", "DenseNet", "EfficientNet", "MobileNet", "ConvNeXt", "VisionTransformer", "SwinTransformer", "UnknownModel")


def detect_architecture_family(model: nn.Module) -> str:
    """Infer family from module graph signatures, not a specific model/version."""
    root_name = type(model).__name__.lower()
    modules = list(model.named_modules())
    names = " ".join(name.lower() for name, _ in modules)
    classes = " ".join(type(module).__name__.lower() for _, module in modules)
    if "swin" in root_name or "swintransformerblock" in classes:
        return "SwinTransformer"
    if "visiontransformer" in root_name or ("encoderblock" in classes and "heads" in names):
        return "VisionTransformer"
    if "convnext" in root_name or "cnblock" in classes:
        return "ConvNeXt"
    if "densenet" in root_name or "_denselayer" in classes or "denseblock" in names:
        return "DenseNet"
    if "efficientnet" in root_name or "mbconv" in classes or "fusedmbconv" in classes:
        return "EfficientNet"
    if "mobilenet" in root_name or "invertedresidual" in classes:
        return "MobileNet"
    if "resnet" in root_name or (("basicblock" in classes or "bottleneck" in classes) and any(name.startswith("layer") for name, _ in modules)):
        return "ResNet"
    return "UnknownModel"
