"""
=========================================================
UAIRE - Universal AI Reliability Engine

Gradient Stability Assessment Module (GSAM)

Extracts:
1. Gradient Mean
2. Gradient Standard Deviation
3. Gradient Maximum
4. Gradient Minimum
5. Gradient Norm
6. Gradient Energy
7. Gradient Sparsity
8. Gradient Entropy
=========================================================
"""

import torch

from .base_extractor import ReliabilitySignalExtractor


class GradientSignalExtractor(ReliabilitySignalExtractor):

    def __init__(self):
        super().__init__("Gradient Signal Extractor")

    def extract(
        self,
        model,
        input_tensor,
        output,
        gradients=None,
        **kwargs
    ):

        if gradients is None:
            raise ValueError("Gradients must be provided.")

        x = gradients.detach().flatten()

        gradient_mean = torch.mean(x)

        gradient_std = torch.std(x)

        gradient_max = torch.max(x)

        gradient_min = torch.min(x)

        gradient_norm = torch.norm(x, p=2)

        gradient_energy = torch.mean(x ** 2)

        gradient_sparsity = torch.mean((torch.abs(x) < 1e-6).float())

        probabilities = torch.softmax(torch.abs(x), dim=0)

        gradient_entropy = -torch.sum(
            probabilities * torch.log(probabilities + 1e-12)
        )

        return {

            "gradient_mean": gradient_mean.item(),

            "gradient_std": gradient_std.item(),

            "gradient_max": gradient_max.item(),

            "gradient_min": gradient_min.item(),

            "gradient_norm": gradient_norm.item(),

            "gradient_energy": gradient_energy.item(),

            "gradient_sparsity": gradient_sparsity.item(),

            "gradient_entropy": gradient_entropy.item()
        }