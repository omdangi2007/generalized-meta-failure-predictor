"""Safe explanation result when an advanced backend is unavailable."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import BaseExplainer


class FallbackExplainer(BaseExplainer):
    method_name = "fallback"

    def supports(self, runtime_context, additional_data=None) -> bool:
        return True

    def explain(self, *, model, runtime_context, input_tensor, prediction=None, additional_data=None):
        data = additional_data or {}
        confidence = data.get("confidence")
        chosen = prediction
        warnings = list(data.get("fallback_warnings", ()))
        if isinstance(input_tensor, torch.Tensor) and confidence is None:
            try:
                with torch.no_grad():
                    logits = model(input_tensor.to(next(model.parameters()).device))
                    probs = torch.softmax(logits, dim=1)
                chosen = int(torch.argmax(probs, dim=1).item()) if chosen is None else int(chosen)
                confidence = float(probs[0, chosen].item())
            except Exception as error:
                warnings.append(f"Prediction confidence unavailable: {type(error).__name__}: {error}")
        warnings.append("Advanced explanation unavailable; returning runtime and prediction summary.")
        return {
            "heatmap": None, "importance_scores": {},
            "text_summary": f"{runtime_context.metadata.architecture_family} runtime summary for prediction {chosen if chosen is not None else 'Unknown'}.",
            "confidence": float(confidence) if confidence is not None else None,
            "metadata": {
                "prediction": chosen, "model_class": runtime_context.metadata.model_class,
                "parameters": runtime_context.metadata.total_parameters,
                "feature_layer": runtime_context.hooks.feature_layer,
            },
            "warnings": tuple(warnings),
        }

    def metadata(self) -> dict[str, Any]:
        return {"backend": "runtime summary", "always_available": True}
