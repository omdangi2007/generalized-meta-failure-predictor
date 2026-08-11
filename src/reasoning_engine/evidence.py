"""Conversion of heterogeneous UAIRE outputs into standardized evidence objects."""

from __future__ import annotations

from typing import Any

from .contracts import Evidence


class EvidenceCollector:
    def collect(self, *, signals: dict[str, Any] | None = None, meta_prediction: Any = None, explanation: Any = None, runtime_context=None) -> tuple[Evidence, ...]:
        evidence = []
        for source, values in self._sources(signals or {}, meta_prediction, explanation, runtime_context):
            for feature, value in self._flatten(values).items():
                if isinstance(value, (int, float, bool)):
                    numeric = float(value)
                    importance = self._importance(feature, numeric)
                    evidence.append(Evidence(source, feature, numeric, None, self._severity(importance), 1.0, importance, self._describe(feature, numeric), {}))
        return tuple(evidence)

    def _sources(self, signals, meta_prediction, explanation, runtime):
        yield "signals", signals
        if meta_prediction is not None:
            yield "meta_prediction", meta_prediction if isinstance(meta_prediction, dict) else {"failure_probability": meta_prediction}
        if explanation is not None:
            values = getattr(explanation, "importance_scores", None) or (explanation.get("importance_scores", {}) if isinstance(explanation, dict) else {})
            yield "explainability", values
        if runtime is not None:
            yield "runtime", {"number_of_output_classes": runtime.metadata.number_of_output_classes, "total_parameters": runtime.metadata.total_parameters}

    @staticmethod
    def _flatten(values, prefix=""):
        result = {}
        if isinstance(values, dict):
            for key, value in values.items():
                result.update(EvidenceCollector._flatten(value, f"{prefix}{key}."))
        else:
            result[prefix[:-1]] = values
        return result

    @staticmethod
    def _importance(feature: str, value: float) -> float:
        feature = feature.split(".")[-1]
        if feature in {"failure_probability", "msp_ood_score", "entropy", "gradient_std", "activation_std"}:
            return max(0.0, min(1.0, abs(value)))
        if feature == "confidence":
            return max(0.0, min(1.0, 1.0 - value))
        return min(1.0, abs(value) / (abs(value) + 1.0))

    @staticmethod
    def _severity(importance: float) -> str:
        return "high" if importance >= 0.7 else "medium" if importance >= 0.35 else "low"

    @staticmethod
    def _describe(feature: str, value: float) -> str:
        return f"Observed {feature}={value:.4f}."
