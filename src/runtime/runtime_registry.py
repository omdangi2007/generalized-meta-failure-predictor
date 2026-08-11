"""Family-level runtime strategies and capabilities, independent of model versions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStrategy:
    preprocessing_profile: str
    hook_strategy: str
    supported_explanations: tuple[str, ...]
    supported_ood_methods: tuple[str, ...]
    recommended_image_size: int | None
    supported_signal_extractors: tuple[str, ...]


COMMON_SIGNALS = ("prediction_confidence", "activation_signals", "gradient_signals", "input_quality")
COMMON_OOD = ("msp", "energy", "mahalanobis")
CNN_STRATEGY = RuntimeStrategy("ImageNet", "cnn_last_convolution", ("gradcam", "lime", "shap"), COMMON_OOD, 224, COMMON_SIGNALS)
TRANSFORMER_STRATEGY = RuntimeStrategy("ImageNet", "transformer_last_encoder_block", ("lime", "shap"), COMMON_OOD, 224, COMMON_SIGNALS)
UNKNOWN_STRATEGY = RuntimeStrategy("Custom", "deepest_feature_producing_module", ("lime", "shap"), ("msp", "energy"), None, COMMON_SIGNALS)
RUNTIME_REGISTRY = {
    "ResNet": CNN_STRATEGY, "DenseNet": CNN_STRATEGY, "EfficientNet": CNN_STRATEGY,
    "MobileNet": CNN_STRATEGY, "ConvNeXt": CNN_STRATEGY,
    "VisionTransformer": TRANSFORMER_STRATEGY, "SwinTransformer": TRANSFORMER_STRATEGY,
    "UnknownModel": UNKNOWN_STRATEGY,
}


def get_runtime_strategy(architecture_family: str) -> RuntimeStrategy:
    return RUNTIME_REGISTRY[architecture_family]
