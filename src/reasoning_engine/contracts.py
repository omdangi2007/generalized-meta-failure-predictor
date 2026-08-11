"""Immutable public data contracts for deterministic UAIRE reasoning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    source: str
    feature: str
    value: float | str | bool | None
    threshold: float | None
    severity: str
    confidence: float
    importance: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningNode:
    node_id: str
    node_type: str
    label: str
    conclusion: str
    evidence_indices: tuple[int, ...]
    score: float
    parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningGraph:
    nodes: tuple[ReasoningNode, ...]
    edges: tuple[tuple[str, str], ...]

    def traverse(self) -> tuple[ReasoningNode, ...]:
        return self.nodes

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [asdict(node) for node in self.nodes], "edges": [list(edge) for edge in self.edges], "visualization": {"layout": "evidence-rule-recommendation"}}


@dataclass(frozen=True)
class Recommendation:
    label: str
    score: float
    confidence: float
    reason: str
    priority: str


@dataclass(frozen=True)
class ReasoningResult:
    evidence: tuple[Evidence, ...]
    graph: ReasoningGraph
    natural_language_explanation: str
    recommendation: Recommendation
    confidence: float
    warnings: tuple[str, ...]
    metadata: dict[str, Any]
    runtime: Any
