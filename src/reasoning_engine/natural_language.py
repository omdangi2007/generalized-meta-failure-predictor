"""Deterministic template-based explanations from a reasoning graph."""

from __future__ import annotations

from .contracts import Recommendation, ReasoningNode


class NaturalLanguageGenerator:
    def generate(self, nodes: tuple[ReasoningNode, ...], recommendation: Recommendation) -> str:
        if not nodes:
            return "The prediction exhibits no material reliability indicators. Overall evidence supports trusting the prediction within the configured operating assumptions."
        conclusions = " ".join(node.conclusion for node in nodes[:4])
        return (
            f"The prediction exhibits {len(nodes)} reliability indicator{'s' if len(nodes) != 1 else ''}. "
            f"{conclusions} Overall evidence recommends: {recommendation.label}."
        )
