"""Universal architecture-aware explainability orchestration for UAIRE."""

from .contracts import BaseExplainer
from .engine import ExplainabilityEngine
from .registry import ExplainabilityRegistry
from .report import ExplanationResult, report_payload, write_explanation_report
from .selector import ArchitectureSelector

__all__ = [
    "ArchitectureSelector", "BaseExplainer", "ExplainabilityEngine", "ExplainabilityRegistry",
    "ExplanationResult", "report_payload", "write_explanation_report",
]
