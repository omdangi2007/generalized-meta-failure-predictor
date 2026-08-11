# UAIRE Universal Explainability Engine

`src.explainability` is an additive architecture-aware explanation layer. Callers provide a model, its existing `RuntimeContext`, input, and optional supplemental data; they never select GradCAM, SHAP, LIME, or attention rollout directly.

```python
from src.explainability import ExplainabilityEngine, write_explanation_report

result = ExplainabilityEngine().explain(
    model=model,
    runtime_context=context,
    input_tensor=image_tensor,
    prediction=prediction,
    additional_data={"image": original_pil_image},
)
write_explanation_report(result, "results/explanations")
```

## Architecture

```text
PyTorch Model → RuntimeContext → ArchitectureSelector → ExplainabilityRegistry
                                                        ↓
                                                Chosen Explainer
                                                        ↓
                                              ExplanationResult → Report Generator
```

## Selection flow

- Tabular/meta-model request with feature data: SHAP wrapper.
- Runtime-discovered transformer blocks: Attention Rollout placeholder.
- Runtime capability supports GradCAM: existing GradCAM wrapper.
- LIME is used automatically when GradCAM is unavailable but runtime capabilities support image-superpixel explanation.
- Otherwise, or when an advanced backend fails: fallback runtime/prediction summary.

Selection only consults `RuntimeContext` and supplied data; no model-version or backbone-specific branches are used.

## Registered explainers

- `gradcam` delegates to `src.xai.gradcam_features.GradCAMExplainer`.
- `shap` delegates to `src.xai.shap_features.SHAPMetaExplainer`.
- `lime` delegates to `src.xai.lime_features.LIMEImageExplainer`.
- `attention_rollout` provides validated final-transformer hook capture and a clearly marked placeholder visualization.
- `fallback` provides prediction, confidence, runtime metadata, and reasons advanced output was unavailable.

`ExplainabilityRegistry` supports `register(explainer)`, `unregister(method_name)`, `get(method_name)`, and `list()` for future implementations. Each explainer implements `BaseExplainer.supports()`, `explain()`, and `metadata()`.

## Reports

`write_explanation_report()` emits JSON and Markdown with selected method, architecture, runtime/hook data, summary, warnings, timing, metadata, scores, and heatmap shape/dtype. Array payloads are represented by metadata instead of embedding large binary data.

## Limitations

- The existing GradCAM and LIME backends retain their own image-preprocessing assumptions.
- Attention rollout is deliberately a replaceable placeholder, not a research-grade attention rollout implementation.
- The existing SHAP backend explains the saved UAIRE meta-model and expects reliability feature data.
