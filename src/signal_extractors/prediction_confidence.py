"""
=========================================================
UAIRE - Universal AI Reliability Engine

Prediction Confidence Signal Extractor

Extracts:
1. Confidence Score
2. Prediction Entropy
3. Prediction Margin
=========================================================
"""

import torch
import torch.nn.functional as F

from .base_extractor import ReliabilitySignalExtractor


class PredictionConfidenceExtractor(ReliabilitySignalExtractor):

    def __init__(self):
        super().__init__("Prediction Confidence Extractor")

    def extract(
        self,
        model,
        input_tensor,
        output,
        **kwargs
    ):

        probabilities = F.softmax(output, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        entropy = -torch.sum(
            probabilities * torch.log(probabilities + 1e-12),
            dim=1
        )

        top2 = torch.topk(probabilities, k=2, dim=1).values

        prediction_margin = top2[:, 0] - top2[:, 1]

        return {
            "confidence": confidence.item(),
            "entropy": entropy.item(),
            "prediction_margin": prediction_margin.item()
        }