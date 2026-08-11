"""Infer a registry family/variant from checkpoint topology without filename rules."""

from __future__ import annotations

import re

import torch

from .exceptions import UnsupportedArchitectureError


def infer_architecture(state_dict: dict[str, torch.Tensor]) -> tuple[str, str | None]:
    keys = tuple(state_dict)
    joined = " ".join(keys)
    if "encoder.layers.encoder_layer_" in joined and "heads.head" in joined:
        return "VisionTransformer", "vit_b_16"
    if any("features." in key and ".blocks." in key for key in keys) and "head.weight" in state_dict:
        return "SwinTransformer", "swin_t"
    if any("features." in key and ".block." in key for key in keys) and "classifier.2.weight" in state_dict:
        return "ConvNeXt", "convnext_tiny"
    if any("denseblock" in key for key in keys) and "classifier.weight" in state_dict:
        return "DenseNet", "densenet121"
    if "fc.weight" in state_dict and any(key.startswith("layer1.") for key in keys):
        stage_depths = []
        for stage in range(1, 5):
            indices = {int(match.group(1)) for key in keys if (match := re.match(rf"layer{stage}\.(\d+)\.", key))}
            stage_depths.append(len(indices))
        return "ResNet", "resnet34" if stage_depths == [3, 4, 6, 3] else "resnet18"
    if "classifier.1.weight" in state_dict:
        if any("block.0." in key for key in keys) and any(key.startswith("features.8.") for key in keys):
            return "EfficientNet", "efficientnet_b0"
        return "MobileNet", "mobilenet_v2"
    raise UnsupportedArchitectureError("Could not infer a supported architecture from checkpoint state_dict keys.")


def normalize_architecture(value: str) -> str:
    compact = value.replace("_", "").replace("-", "").lower()
    compact = re.sub(r"\d+$", "", compact)
    aliases = {
        "resnet": "ResNet", "densenet": "DenseNet", "efficientnet": "EfficientNet", "mobilenet": "MobileNet",
        "convnext": "ConvNeXt", "vit": "VisionTransformer", "visiontransformer": "VisionTransformer",
        "swin": "SwinTransformer", "swintransformer": "SwinTransformer",
    }
    if compact not in aliases:
        raise UnsupportedArchitectureError(f"Unsupported architecture metadata '{value}'.")
    return aliases[compact]
