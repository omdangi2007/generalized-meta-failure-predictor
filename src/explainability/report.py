"""Portable JSON and Markdown reports for common ExplanationResult contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExplanationResult:
    method_name: str
    architecture: str
    heatmap: Any
    importance_scores: dict[str, float | int]
    text_summary: str
    warnings: tuple[str, ...]
    confidence: float | None
    runtime: Any
    metadata: dict[str, Any]


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def report_payload(result: ExplanationResult) -> dict[str, Any]:
    runtime = result.runtime
    return {
        "selected_method": result.method_name,
        "architecture": result.architecture,
        "confidence": result.confidence,
        "summary": result.text_summary,
        "warnings": list(result.warnings),
        "timing_seconds": result.metadata.get("timing_seconds"),
        "runtime": {
            "model_class": runtime.metadata.model_class,
            "image_size": runtime.metadata.image_size,
            "normalization_profile": runtime.preprocessing.name,
            "hook_strategy": runtime.hooks.strategy,
            "feature_layer": runtime.hooks.feature_layer,
        },
        "importance_scores": _json_safe(result.importance_scores),
        "heatmap": _json_safe(result.heatmap),
        "metadata": _json_safe(result.metadata),
    }


def write_explanation_report(result: ExplanationResult, output_dir: str | Path, stem: str = "explanation") -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = report_payload(result)
    json_path = destination / f"{stem}.json"
    markdown_path = destination / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    warnings = "\n".join(f"- {warning}" for warning in result.warnings) or "- None"
    scores = "\n".join(f"- `{name}`: {score}" for name, score in result.importance_scores.items()) or "- Not available"
    markdown_path.write_text(
        f"# UAIRE Explanation Report\n\n"
        f"- Method: `{result.method_name}`\n- Architecture: `{result.architecture}`\n- Confidence: `{result.confidence}`\n"
        f"- Runtime hook: `{result.runtime.hooks.feature_layer}`\n- Timing seconds: `{result.metadata.get('timing_seconds')}`\n\n"
        f"## Summary\n\n{result.text_summary}\n\n## Importance Scores\n\n{scores}\n\n## Warnings\n\n{warnings}\n",
        encoding="utf-8",
    )
    return json_path, markdown_path
