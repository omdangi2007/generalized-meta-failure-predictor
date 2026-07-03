"""
=========================================================
UAIRE

Energy-Based OOD Detector

Returns:
1. Energy Score
2. Normalized Energy Score

Reference:
Energy-based Out-of-Distribution Detection (NeurIPS 2020)
=========================================================
"""

import torch

from .base_detector import OODDetector


class EnergyDetector(OODDetector):

    def __init__(self):

        super().__init__("Energy Detector")

    def detect(
        self,
        model,
        input_tensor,
        output,
        temperature=1.0,
        **kwargs
    ):

        # Energy Score
        energy = -temperature * torch.logsumexp(
            output / temperature,
            dim=1
        )

        energy = energy.item()

        # Normalized Energy
        normalized_energy = torch.sigmoid(
            torch.tensor(-energy)
        ).item()

        return {

            "energy_score": float(energy),

            "normalized_energy": float(normalized_energy)

        }