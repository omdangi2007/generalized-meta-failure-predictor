"""Common interfaces for architecture-independent explanation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.runtime import RuntimeContext


class BaseExplainer(ABC):
    method_name: str

    @abstractmethod
    def supports(self, runtime_context: RuntimeContext, additional_data: dict[str, Any] | None = None) -> bool:
        pass

    @abstractmethod
    def explain(self, *, model, runtime_context: RuntimeContext, input_tensor, prediction=None, additional_data: dict[str, Any] | None = None):
        pass

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        pass
