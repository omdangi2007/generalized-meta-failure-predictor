"""Select explanation strategy solely from RuntimeContext capabilities and input type."""

from __future__ import annotations


class ArchitectureSelector:
    def select(self, runtime_context, registry, additional_data=None) -> str:
        data = additional_data or {}
        if (data.get("meta_model") or data.get("tabular") or data.get("features") is not None) and self._supported("shap", runtime_context, registry, data):
            return "shap"
        if runtime_context.metadata.transformer_blocks and self._supported("attention_rollout", runtime_context, registry, data):
            return "attention_rollout"
        if self._supported("gradcam", runtime_context, registry, data):
            return "gradcam"
        if self._supported("lime", runtime_context, registry, data):
            return "lime"
        return "fallback"

    @staticmethod
    def _supported(method_name, runtime_context, registry, data) -> bool:
        explainer = registry.get(method_name)
        return bool(explainer and explainer.supports(runtime_context, data))
