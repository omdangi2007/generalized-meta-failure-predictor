# UAIRE Universal Runtime Engine

`src.runtime` is an additive runtime abstraction. It inspects an arbitrary PyTorch `nn.Module` and returns one immutable `RuntimeContext`; it does not modify the adapter, inference predictor, dataset pipeline, model training, Streamlit UI, signal extractors, or saved meta-model.

```python
from src.runtime import RuntimeEngine

context = RuntimeEngine().create_context(model)
print(context.metadata.architecture_family)
print(context.hooks.activation_layer)
```

## Runtime flow

```text
PyTorch model
  → topology inspection
  → architecture-family detection
  → registry strategy and preprocessing selection
  → hook selection
  → validation
  → immutable RuntimeContext
```

`RuntimeContext` contains immutable `RuntimeMetadata`, `HookConfiguration`, `PreprocessingProfile`, and `SupportedCapabilities`, plus the original model reference. Metadata covers parameter counts, classifier output classes, feature extractor, final convolution, transformer blocks, image size, and normalization-profile selection.

## Families and hooks

| Family | Detection | Hook strategy | Default profile |
|---|---|---|---|
| ResNet, DenseNet, EfficientNet, MobileNet, ConvNeXt | inspected module topology | last `Conv2d` | ImageNet |
| Vision Transformer, Swin Transformer | inspected encoder/block topology | final transformer block | ImageNet / 224 |
| UnknownModel | no supported topology signature | deepest non-classifier feature module | explicit Custom required |

The registry also exposes supported explanation methods, OOD methods, signal extractors, and recommended input size. A default profile is a runtime strategy default, not evidence of the checkpoint’s original training preprocessing; callers can supply a specific named or `custom_profile(...)` profile.

## Validation

The engine rejects unknown topologies by default, missing classifier heads, no usable feature extractor, missing hook layers, and malformed/custom-unspecified profiles. Use `allow_unknown=True` only with an explicit caller-supplied `Custom` profile.

## Limitations

- Architecture recognition covers common torchvision-style topology signatures, not every third-party implementation.
- Checkpoints typically do not contain reliable dataset normalization metadata; registry defaults may require caller override.
- The current protected inference predictor is intentionally not migrated to consume `RuntimeContext` in this phase.
- Transformer activation shapes may require explanation implementations that explicitly support token tensors.
