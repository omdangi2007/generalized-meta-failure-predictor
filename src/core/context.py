"""
=========================================================
UAIRE Core

Reliability Context

Carries all information related to a single prediction.
=========================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ReliabilityContext:

    # Model Information
    model: Any = None
    model_name: str = ""

    # Dataset Information
    dataset_name: str = ""

    # Input
    input_tensor: Any = None

    # Output
    logits: Any = None
    probabilities: Any = None
    prediction: Any = None
    target: Any = None

    # Internal Network State
    activations: Dict[str, Any] = field(default_factory=dict)
    gradients: Dict[str, Any] = field(default_factory=dict)

    # Extracted Reliability Signals
    signals: Dict[str, Any] = field(default_factory=dict)

    # Additional Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_signal(self, name, value):
        self.signals[name] = value

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def get_signal(self, name):
        return self.signals.get(name)

    def get_metadata(self, key):
        return self.metadata.get(key)