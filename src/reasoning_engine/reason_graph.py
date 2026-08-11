"""Evidence-to-rule-to-recommendation graph construction and serialization metadata."""

from __future__ import annotations

from .contracts import ReasoningGraph, ReasoningNode


class ReasonGraphBuilder:
    def build(self, rule_nodes: tuple[ReasoningNode, ...], recommendation_label: str, recommendation_score: float) -> ReasoningGraph:
        recommendation = ReasoningNode(
            node_id="recommendation", node_type="recommendation", label=recommendation_label,
            conclusion="Final trust recommendation.", evidence_indices=(), score=recommendation_score,
            parents=tuple(node.node_id for node in rule_nodes),
        )
        edges = []
        for node in rule_nodes:
            edges.extend((parent, node.node_id) for parent in node.parents)
            edges.append((node.node_id, recommendation.node_id))
        return ReasoningGraph(nodes=(*rule_nodes, recommendation), edges=tuple(edges))
