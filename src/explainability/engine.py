"""One public architecture-aware entry point for UAIRE explanation selection."""

from __future__ import annotations

import time

from .registry import ExplainabilityRegistry
from .selector import ArchitectureSelector


class ExplainabilityEngine:
    def __init__(self, registry: ExplainabilityRegistry | None = None, selector: ArchitectureSelector | None = None):
        self.registry = registry or ExplainabilityRegistry()
        self.selector = selector or ArchitectureSelector()

    def explain(self, *, model, runtime_context, input_tensor, prediction=None, additional_data=None):
        from .report import ExplanationResult

        data = dict(additional_data or {})
        selected = self.selector.select(runtime_context, self.registry, data)
        explainer = self.registry.get(selected)
        started = time.perf_counter()
        try:
            payload = explainer.explain(model=model, runtime_context=runtime_context, input_tensor=input_tensor, prediction=prediction, additional_data=data)
        except Exception as error:
            if selected == "fallback":
                raise
            fallback = self.registry.get("fallback")
            data["fallback_warnings"] = (f"{selected} backend failed: {type(error).__name__}: {error}",)
            payload = fallback.explain(model=model, runtime_context=runtime_context, input_tensor=input_tensor, prediction=prediction, additional_data=data)
            selected, explainer = "fallback", fallback
        elapsed = time.perf_counter() - started
        warnings = tuple(runtime_context.capabilities.warnings) + tuple(payload.get("warnings", ()))
        return ExplanationResult(
            method_name=selected, architecture=runtime_context.metadata.architecture_family,
            heatmap=payload.get("heatmap"), importance_scores=payload.get("importance_scores", {}),
            text_summary=payload.get("text_summary", ""), warnings=warnings,
            confidence=payload.get("confidence"), runtime=runtime_context,
            metadata={**explainer.metadata(), **payload.get("metadata", {}), "timing_seconds": elapsed},
        )
