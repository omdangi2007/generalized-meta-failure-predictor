"""
=========================================================
UAIRE - Universal AI Reliability Engine

Neural Activation Health Assessment Module (NAHAM)

Extracts:
1. Activation Mean
2. Activation Standard Deviation
3. Activation Maximum
4. Activation Minimum
5. Activation Energy
6. Activation Sparsity
7. Dead Neuron Ratio
8. Positive Activation Ratio
9. Activation Entropy
=========================================================
"""

import torch

from .base_extractor import ReliabilitySignalExtractor


class ActivationSignalExtractor(ReliabilitySignalExtractor):

    def __init__(self):
        super().__init__("Activation Signal Extractor")

    def extract(
        self,
        model,
        input_tensor,
        output,
        activations=None,
        **kwargs
    ):

        if activations is None:
            raise ValueError("Activations must be provided.")

        x = activations.detach().flatten()

        activation_mean = torch.mean(x)

        activation_std = torch.std(x)

        activation_max = torch.max(x)

        activation_min = torch.min(x)

        activation_energy = torch.mean(x ** 2)

        activation_sparsity = torch.mean((x == 0).float())

        dead_neuron_ratio = torch.mean((x <= 0).float())

        positive_activation_ratio = torch.mean((x > 0).float())

        probabilities = torch.softmax(x, dim=0)

        activation_entropy = -torch.sum(
            probabilities * torch.log(probabilities + 1e-12)
        )

        return {

            "activation_mean": activation_mean.item(),

            "activation_std": activation_std.item(),

            "activation_max": activation_max.item(),

            "activation_min": activation_min.item(),

            "activation_energy": activation_energy.item(),

            "activation_sparsity": activation_sparsity.item(),

            "dead_neuron_ratio": dead_neuron_ratio.item(),

            "positive_activation_ratio": positive_activation_ratio.item(),

            "activation_entropy": activation_entropy.item()
        }