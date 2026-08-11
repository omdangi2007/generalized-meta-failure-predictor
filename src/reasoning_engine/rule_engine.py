"""Deterministic rule-only reliability reasoning; no external language model."""

from __future__ import annotations

from dataclasses import replace

from .contracts import Evidence, ReasoningNode
from .registry import ReasoningRegistry, default_registry


class RuleEngine:
    def __init__(self, registry: ReasoningRegistry | None = None):
        self.registry = registry or default_registry()

    def evaluate(self, evidence: tuple[Evidence, ...]) -> tuple[tuple[Evidence, ...], tuple[ReasoningNode, ...]]:
        updated = list(evidence)
        nodes = []
        for rule in self.registry.list():
            indices = tuple(index for index, item in enumerate(updated) if item.feature.split(".")[-1] == rule.feature and isinstance(item.value, (int, float)) and rule.predicate(float(item.value), rule.threshold))
            if not indices:
                continue
            for index in indices:
                item = updated[index]
                updated[index] = replace(item, threshold=rule.threshold, importance=max(item.importance, rule.score), severity="high" if rule.score >= 0.7 else item.severity)
            nodes.append(ReasoningNode(
                node_id=f"rule:{rule.name}", node_type="rule", label=rule.name,
                conclusion=rule.conclusion, evidence_indices=indices, score=rule.score,
                parents=tuple(f"evidence:{index}" for index in indices),
            ))
        return tuple(updated), tuple(nodes)
