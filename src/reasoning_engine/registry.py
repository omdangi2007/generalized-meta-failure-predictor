"""Extensible registry for deterministic reasoning rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RuleDefinition:
    name: str
    feature: str
    threshold: float
    predicate: Callable[[float, float], bool]
    conclusion: str
    score: float


class ReasoningRegistry:
    def __init__(self):
        self._rules = {}

    def register(self, rule: RuleDefinition) -> None:
        self._rules[rule.name] = rule

    def unregister(self, name: str) -> None:
        self._rules.pop(name, None)

    def list(self) -> tuple[RuleDefinition, ...]:
        return tuple(self._rules[name] for name in sorted(self._rules))


def default_registry() -> ReasoningRegistry:
    registry = ReasoningRegistry()
    for rule in (
        RuleDefinition("LowConfidence", "confidence", 0.50, lambda value, threshold: value < threshold, "Model confidence is below the expected operating threshold.", 0.70),
        RuleDefinition("HighUncertainty", "entropy", 1.00, lambda value, threshold: value > threshold, "Prediction entropy indicates elevated uncertainty.", 0.60),
        RuleDefinition("DistributionShift", "msp_ood_score", 0.40, lambda value, threshold: value > threshold, "OOD evidence indicates possible distribution shift.", 0.75),
        RuleDefinition("LikelyFailure", "failure_probability", 0.50, lambda value, threshold: value >= threshold, "The meta prediction indicates elevated failure risk.", 0.90),
        RuleDefinition("PoorImageQuality", "laplacian_variance", 20.0, lambda value, threshold: value < threshold, "Low image-detail variance suggests a blurred or low-detail input.", 0.45),
        RuleDefinition("InternalRepresentationUnstable", "activation_std", 1.50, lambda value, threshold: value > threshold, "Activation variability suggests unstable internal representations.", 0.55),
        RuleDefinition("GradientInstability", "gradient_std", 0.50, lambda value, threshold: value > threshold, "Gradient variability suggests unstable sensitivity around the prediction.", 0.55),
    ):
        registry.register(rule)
    return registry
