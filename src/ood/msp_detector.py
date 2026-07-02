"""
=========================================================
UAIRE

Maximum Softmax Probability (MSP)

Returns:
1. Maximum Softmax Probability
2. OOD Score
=========================================================
"""

import torch

from .base_detector import OODDetector


class MSPDetector(OODDetector):

    def __init__(self):

        super().__init__("MSP Detector")

    def detect(
        self,
        model,
        input_tensor,
        output,
        **kwargs
    ):

        probabilities = torch.softmax(output, dim=1)

        max_probability = probabilities.max().item()

        ood_score = 1.0 - max_probability

        return {

            "msp_probability": float(max_probability),

            "msp_ood_score": float(ood_score)

        }