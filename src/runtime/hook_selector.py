"""Architecture-aware selection of feature, GradCAM, activation, and gradient hooks."""

from __future__ import annotations

import torch.nn as nn

from .runtime_metadata import HookConfiguration, RuntimeMetadata


class HookSelector:
    def select(self, model: nn.Module, metadata: RuntimeMetadata) -> HookConfiguration:
        names = dict(model.named_modules())
        if metadata.architecture_family in {"ResNet", "DenseNet", "EfficientNet", "MobileNet", "ConvNeXt"}:
            return self._configuration(metadata.last_convolution_layer, "cnn_last_convolution")
        if metadata.architecture_family in {"VisionTransformer", "SwinTransformer"}:
            layer = metadata.transformer_blocks[-1] if metadata.transformer_blocks else metadata.feature_extractor
            return self._configuration(layer, "transformer_last_encoder_block")
        return self._configuration(self._deepest_feature_layer(names, metadata.classifier_head), "deepest_feature_producing_module")

    @staticmethod
    def _configuration(layer: str | None, strategy: str) -> HookConfiguration:
        return HookConfiguration(layer, layer, layer, layer, strategy)

    @staticmethod
    def _deepest_feature_layer(modules: dict[str, nn.Module], classifier: str | None) -> str | None:
        for name, module in reversed(list(modules.items())):
            if not name or name == classifier or isinstance(module, nn.Linear):
                continue
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LayerNorm, nn.BatchNorm2d)) or not list(module.children()):
                return name
        return None
