"""Single-sample bridge from the unchanged UAIRE adapter/pipeline to a row."""

from __future__ import annotations

import numpy as np
import torch

from src.adapters.base_adapter import UniversalModelAdapter
from src.core.universal_reliability_pipeline import UniversalReliabilityPipeline
from .schema import UAIRE_FEATURE_COLUMNS


class UAIREFeatureExtractor:
    def __init__(self, model, device: str):
        self.adapter = UniversalModelAdapter(model=model, device=device)
        self.pipeline = UniversalReliabilityPipeline()

    def extract(self, input_tensor: torch.Tensor, original_image) -> tuple[int, dict[str, object]]:
        # Existing extractors use scalar .item() calls; one-sample collection is
        # therefore required to preserve their current definitions exactly.
        collected = self.adapter.collect(input_tensor.unsqueeze(0), compute_gradients=True)
        features = self.pipeline.extract(
            model=self.adapter.model,
            input_tensor=input_tensor.unsqueeze(0).to(self.adapter.device),
            output=collected["logits"],
            activations=collected["activations"],
            gradients=collected["gradients"],
            image=np.asarray(original_image),
        )
        unknown = set(features) - set(UAIRE_FEATURE_COLUMNS)
        missing = set(UAIRE_FEATURE_COLUMNS) - set(features)
        if unknown or missing:
            raise RuntimeError(f"UAIRE feature contract mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        return int(collected["prediction"].item()), {name: features[name] for name in UAIRE_FEATURE_COLUMNS}

    def cleanup(self) -> None:
        self.adapter.cleanup()
