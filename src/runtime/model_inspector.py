"""Model graph inspection used to construct immutable runtime metadata."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .architecture_detector import detect_architecture_family
from .runtime_metadata import RuntimeMetadata


class ModelInspector:
    def inspect(self, model: nn.Module, normalization_profile: str | None = None, image_size: int | None = None) -> RuntimeMetadata:
        if not isinstance(model, nn.Module):
            raise TypeError("Runtime inspection requires torch.nn.Module.")
        modules = list(model.named_modules())
        named = {name: module for name, module in modules}
        classifier = self._classifier_head(modules)
        classifier_module = named.get(classifier) if classifier else None
        output_classes = classifier_module.out_features if isinstance(classifier_module, nn.Linear) else None
        convolutions = [name for name, module in modules if isinstance(module, nn.Conv2d)]
        transformer_blocks = tuple(name for name, module in modules if self._is_transformer_block(module))
        family = detect_architecture_family(model)
        warnings = []
        if classifier is None:
            warnings.append("No nn.Linear classifier head was discovered.")
        if not convolutions and not transformer_blocks:
            warnings.append("No convolution or transformer feature-producing block was discovered.")
        return RuntimeMetadata(
            architecture_family=family,
            model_class=f"{type(model).__module__}.{type(model).__name__}",
            total_parameters=sum(parameter.numel() for parameter in model.parameters()),
            trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            number_of_output_classes=output_classes,
            classifier_head=classifier,
            feature_extractor=self._feature_extractor(modules, family),
            last_convolution_layer=convolutions[-1] if convolutions else None,
            transformer_blocks=transformer_blocks,
            image_size=image_size if image_size is not None else self._image_size(model),
            normalization_profile=normalization_profile,
            inspection_warnings=tuple(warnings),
        )

    @staticmethod
    def _classifier_head(modules: list[tuple[str, nn.Module]]) -> Optional[str]:
        for name, module in reversed(modules):
            if name and isinstance(module, nn.Linear):
                return name
        return None

    @staticmethod
    def _feature_extractor(modules: list[tuple[str, nn.Module]], family: str) -> Optional[str]:
        preferred = {
            "ResNet": ("layer4",), "DenseNet": ("features",), "EfficientNet": ("features",),
            "MobileNet": ("features",), "ConvNeXt": ("features",), "VisionTransformer": ("encoder",),
            "SwinTransformer": ("features",),
        }.get(family, ())
        available = {name for name, _ in modules}
        for name in preferred:
            if name in available:
                return name
        for name, module in reversed(modules):
            if name and not isinstance(module, nn.Linear) and len(list(module.children())):
                return name
        return None

    @staticmethod
    def _is_transformer_block(module: nn.Module) -> bool:
        name = type(module).__name__.lower()
        return ("transformer" in name and "block" in name) or name in {"encoderblock", "swintransformerblock"}

    @staticmethod
    def _image_size(model: nn.Module) -> Optional[int]:
        image_size = getattr(model, "image_size", None)
        return image_size if isinstance(image_size, int) and image_size > 0 else None
