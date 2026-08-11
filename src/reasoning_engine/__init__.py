"""Deterministic, architecture-independent reliability reasoning for UAIRE."""

from .contracts import Evidence, Recommendation, ReasoningGraph, ReasoningNode, ReasoningResult
from .engine import ReasoningEngine
from .registry import ReasoningRegistry, RuleDefinition
from .report import report_payload, write_reasoning_report

__all__ = [
    "Evidence", "Recommendation", "ReasoningEngine", "ReasoningGraph", "ReasoningNode",
    "ReasoningRegistry", "ReasoningResult", "RuleDefinition", "report_payload", "write_reasoning_report",
]
