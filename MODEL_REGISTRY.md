# UAIRE Universal Model Registry

`src.model_registry` is an additive, plug-and-play checkpoint loading interface. It reconstructs supported models from checkpoint topology, validates compatibility, then creates the existing `RuntimeContext` without changing the RuntimeEngine, predictor, UI, adapters, datasets, training, extractors, or OOD methods.

```python
from src.model_registry import load_model

loaded = load_model("models/resnet18_cifar10.pth")
model = loaded.model
runtime = loaded.runtime_context
```

## Workflow

```text
checkpoint path
  → safe payload classification (.pt/.pth/.ckpt)
  → metadata recovery + state_dict topology inference
  → registry entry / constructor selection
  → model reconstruction
  → key, shape, classifier, and runtime compatibility validation
  → weight loading
  → RuntimeEngine.create_context
  → LoadedModel
```

## Registry entries

ResNet, DenseNet, EfficientNet, MobileNet, ConvNeXt, Vision Transformer, and Swin Transformer are supported. Each registry entry declares constructor, formats, classifier convention, RuntimeEngine compatibility, explanation methods, preprocessing profiles, OOD methods, and minimum torch version. `UnknownModel` is deliberately not reconstructed from bare weights because no safe architecture factory can be inferred.

Metadata recovery is best effort: architecture, variant, dataset, input size, class count, preprocessing profile, checkpoint version, author, and timestamp are read from conventional checkpoint/config fields. Missing fields return `Unknown` rather than failing.

State-dict loading is safe by default. Full serialized models require `allow_unsafe_full_model=True` and must come from a trusted source because PyTorch pickle deserialization is unsafe for untrusted artifacts.

## Limitations

- Variant inference supports the common torchvision layouts currently registered; unusual/custom heads need explicit registry extension.
- A bare state_dict cannot reliably reveal dataset normalization, author, or training provenance.
- `strict=False` can intentionally load only matching state tensors; its compatibility report must be inspected by the caller.
