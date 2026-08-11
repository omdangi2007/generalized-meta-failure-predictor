"""Extensible registry of explanation backends."""

from __future__ import annotations

from .attention_rollout import AttentionRolloutExplainer
from .explanation import GradCAMExplainer, LIMEExplainer, SHAPExplainer
from .fallback import FallbackExplainer


class ExplainabilityRegistry:
    def __init__(self):
        self._explainers = {}
        for explainer in (GradCAMExplainer(), SHAPExplainer(), LIMEExplainer(), AttentionRolloutExplainer(), FallbackExplainer()):
            self.register(explainer)

    def register(self, explainer) -> None:
        if not getattr(explainer, "method_name", None):
            raise ValueError("Explainer must expose a non-empty method_name.")
        self._explainers[explainer.method_name] = explainer

    def unregister(self, method_name: str) -> None:
        if method_name == "fallback":
            raise ValueError("Fallback explainer cannot be unregistered.")
        self._explainers.pop(method_name, None)

    def get(self, method_name: str):
        return self._explainers.get(method_name)

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._explainers))
