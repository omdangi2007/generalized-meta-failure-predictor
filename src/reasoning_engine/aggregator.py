"""Evidence deduplication, normalization, clustering, and ranking."""

from __future__ import annotations

from dataclasses import replace

from .contracts import Evidence


class EvidenceAggregator:
    def __init__(self, minimum_importance: float = 0.05):
        self.minimum_importance = minimum_importance

    def aggregate(self, evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
        merged = {}
        for item in evidence:
            key = (item.source, item.feature)
            if key not in merged or item.importance > merged[key].importance:
                merged[key] = item
            else:
                prior = merged[key]
                merged[key] = replace(prior, metadata={**prior.metadata, "duplicate_count": prior.metadata.get("duplicate_count", 1) + 1})
        kept = [self._normalize(item) for item in merged.values() if item.importance >= self.minimum_importance]
        return tuple(sorted(kept, key=lambda item: item.importance * item.confidence, reverse=True))

    @staticmethod
    def cluster(evidence: tuple[Evidence, ...]) -> dict[str, tuple[Evidence, ...]]:
        clusters = {}
        for item in evidence:
            feature = item.feature.split(".")[-1]
            if feature in {"confidence", "entropy", "prediction_margin"}:
                key = "prediction_uncertainty"
            elif "ood" in feature or "energy" in feature or "mahalanobis" in feature:
                key = "distribution_shift"
            elif feature in {"blur_score", "laplacian_variance", "brightness", "contrast", "edge_density"}:
                key = "input_quality"
            elif "activation" in feature or "gradient" in feature:
                key = "internal_state"
            elif "failure" in feature:
                key = "meta_reliability"
            else:
                key = "other"
            clusters[key] = (*clusters.get(key, ()), item)
        return clusters

    @staticmethod
    def _normalize(item: Evidence) -> Evidence:
        return replace(item, confidence=max(0.0, min(1.0, item.confidence)), importance=max(0.0, min(1.0, item.importance)))
