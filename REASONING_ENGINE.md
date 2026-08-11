# UAIRE Universal Reasoning Engine

`src.reasoning_engine` converts existing reliability outputs into deterministic evidence, rule conclusions, trust recommendations, and readable reports. It does not change model, adapter, extractor, OOD, explainability, or inference calculations.

```python
from src.reasoning_engine import ReasoningEngine, write_reasoning_report

result = ReasoningEngine().reason(
    runtime_context=context,
    signals=reliability_signals,
    meta_prediction={"failure_probability": 0.72},
    explanation=explanation_result,
)
write_reasoning_report(result, "results/reasoning")
```

## Architecture

```text
Reliability signals → EvidenceCollector → EvidenceAggregator → RuleEngine
                                                            ↓
Evidence → Rule → Intermediate conclusion → Recommendation → NaturalLanguageGenerator → Report
```

## Evidence model

Each immutable `Evidence` object records source, feature, value, rule threshold, severity, confidence, importance, description, and metadata. The collector accepts flat or nested signal dictionaries, meta prediction, explainability importance scores, and runtime metadata.

## Rule engine

The default deterministic rules produce LowConfidence, HighUncertainty, DistributionShift, LikelyFailure, PoorImageQuality, InternalRepresentationUnstable, and GradientInstability conclusions. Rules are registered through `ReasoningRegistry`, so future rules can be added without changing the engine. No external language model is used.

## Recommendation system

Rule evidence is aggregated into one of: Trust, Caution, Needs Human Review, Do Not Trust, or Reject Prediction. Each recommendation has label, score, confidence, reason, and priority.

## Extension guide

Create a `RuleDefinition` and register it with a `ReasoningRegistry`; use that registry in `RuleEngine`. New evidence sources should emit standardized `Evidence` objects through an additive collector extension.

## Reports

`write_reasoning_report()` writes `reasoning.json` and `reasoning.md`, including evidence, triggered rules, serialized graph, recommendation, confidence, warnings, summary, and runtime metadata.
