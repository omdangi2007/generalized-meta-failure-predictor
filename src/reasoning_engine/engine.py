"""Public deterministic evidence-to-recommendation UAIRE reasoning pipeline."""

from __future__ import annotations

from .aggregator import EvidenceAggregator
from .contracts import ReasoningResult
from .evidence import EvidenceCollector
from .natural_language import NaturalLanguageGenerator
from .reason_graph import ReasonGraphBuilder
from .recommendation import RecommendationEngine
from .rule_engine import RuleEngine


class ReasoningEngine:
    def __init__(self, collector=None, aggregator=None, rule_engine=None, graph_builder=None, recommendation_engine=None, natural_language=None):
        self.collector = collector or EvidenceCollector()
        self.aggregator = aggregator or EvidenceAggregator()
        self.rule_engine = rule_engine or RuleEngine()
        self.graph_builder = graph_builder or ReasonGraphBuilder()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.natural_language = natural_language or NaturalLanguageGenerator()

    def reason(self, *, runtime_context, signals=None, meta_prediction=None, explanation=None) -> ReasoningResult:
        collected = self.collector.collect(signals=signals, meta_prediction=meta_prediction, explanation=explanation, runtime_context=runtime_context)
        aggregated = self.aggregator.aggregate(collected)
        evidence, nodes = self.rule_engine.evaluate(aggregated)
        recommendation = self.recommendation_engine.recommend(nodes)
        graph = self.graph_builder.build(nodes, recommendation.label, recommendation.score)
        warnings = list(runtime_context.capabilities.warnings)
        if not nodes:
            warnings.append("No configured reliability rule was triggered.")
        if meta_prediction is None:
            warnings.append("No meta prediction was supplied; recommendation excludes meta-model failure evidence.")
        return ReasoningResult(
            evidence=evidence, graph=graph,
            natural_language_explanation=self.natural_language.generate(nodes, recommendation),
            recommendation=recommendation, confidence=recommendation.confidence,
            warnings=tuple(warnings),
            metadata={"evidence_count": len(evidence), "triggered_rules": [node.label for node in nodes], "clusters": {name: len(items) for name, items in self.aggregator.cluster(evidence).items()}},
            runtime=runtime_context,
        )
