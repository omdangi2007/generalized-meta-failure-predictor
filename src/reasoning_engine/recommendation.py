"""Deterministic trust recommendation derived from rule evidence."""

from __future__ import annotations

from .contracts import Recommendation, ReasoningNode


class RecommendationEngine:
    def recommend(self, nodes: tuple[ReasoningNode, ...]) -> Recommendation:
        scores = sorted((node.score for node in nodes), reverse=True)
        risk = min(1.0, sum(scores[:3]) / 2.0) if scores else 0.0
        confidence = min(1.0, 0.45 + 0.12 * len(nodes))
        if risk >= 0.85:
            return Recommendation("Reject Prediction", risk, confidence, "Multiple severe reliability rules were triggered.", "critical")
        if risk >= 0.70:
            return Recommendation("Do Not Trust", risk, confidence, "Strong evidence indicates elevated prediction risk.", "critical")
        if risk >= 0.45:
            return Recommendation("Needs Human Review", risk, confidence, "Several reliability indicators require human review.", "high")
        if risk >= 0.25:
            return Recommendation("Caution", risk, confidence, "Some reliability indicators warrant cautious use.", "medium")
        return Recommendation("Trust", risk, confidence, "No material reliability rule was triggered.", "low")
