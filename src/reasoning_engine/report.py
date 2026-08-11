"""JSON and Markdown export for deterministic reasoning results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .contracts import ReasoningResult


def report_payload(result: ReasoningResult) -> dict:
    return {
        "evidence": [asdict(item) for item in result.evidence],
        "triggered_rules": result.metadata.get("triggered_rules", []),
        "reasoning_graph": result.graph.to_dict(),
        "recommendation": asdict(result.recommendation),
        "warnings": list(result.warnings), "confidence": result.confidence,
        "summary": result.natural_language_explanation, "metadata": result.metadata,
        "runtime": {"architecture": result.runtime.metadata.architecture_family, "model_class": result.runtime.metadata.model_class, "hook_strategy": result.runtime.hooks.strategy},
    }


def write_reasoning_report(result: ReasoningResult, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = report_payload(result)
    json_path, markdown_path = output / "reasoning.json", output / "reasoning.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = "\n".join(f"- `{item.feature}` ({item.source}): {item.value}; severity={item.severity}" for item in result.evidence) or "- None"
    rules = "\n".join(f"- {node.label}: {node.conclusion}" for node in result.graph.nodes if node.node_type == "rule") or "- None"
    warnings = "\n".join(f"- {item}" for item in result.warnings) or "- None"
    markdown_path.write_text(
        f"# UAIRE Reasoning Report\n\n## Recommendation\n\n- Label: **{result.recommendation.label}**\n- Score: {result.recommendation.score:.3f}\n- Confidence: {result.confidence:.3f}\n- Priority: {result.recommendation.priority}\n\n## Summary\n\n{result.natural_language_explanation}\n\n## Evidence\n\n{evidence}\n\n## Triggered Rules\n\n{rules}\n\n## Warnings\n\n{warnings}\n",
        encoding="utf-8",
    )
    return json_path, markdown_path
