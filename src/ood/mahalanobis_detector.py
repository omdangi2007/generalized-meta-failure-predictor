"""
=========================================================
UAIRE

Mahalanobis Distance OOD Detector

Computes the Mahalanobis distance between a sample's
deep feature representation and the training feature
distribution.

Statistics required:
- Mean Feature Vector
- Inverse Covariance Matrix

Reference:
Lee et al., "A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks",
NeurIPS 2018.
=========================================================
"""

import os
import numpy as np
import torch

from .base_detector import OODDetector


class MahalanobisDetector(OODDetector):

    def __init__(
        self,
        stats_path="../models/mahalanobis_stats.npz"
    ):

        super().__init__("Mahalanobis Detector")

        self.stats_loaded = False

        if os.path.exists(stats_path):

            stats = np.load(stats_path)

            self.mean = torch.tensor(
                stats["mean"],
                dtype=torch.float32
            )

            self.inv_cov = torch.tensor(
                stats["inv_cov"],
                dtype=torch.float32
            )

            self.stats_loaded = True

    def detect(
        self,
        model,
        input_tensor,
        output,
        activations=None,
        **kwargs
    ):

        # ----------------------------------------------------
        # Statistics not available
        # ----------------------------------------------------

        if not self.stats_loaded:

            return {

                "mahalanobis_distance": -1.0,

                "mahalanobis_available": False

            }

        # ----------------------------------------------------
        # Activations required
        # ----------------------------------------------------

        if activations is None:

            raise ValueError(
                "Activations are required."
            )

        # ----------------------------------------------------
        # Flatten feature vector
        # ----------------------------------------------------

        feature = activations.flatten().float()

        # ----------------------------------------------------
        # Move statistics to same device
        # ----------------------------------------------------

        mean = self.mean.to(feature.device)
        inv_cov = self.inv_cov.to(feature.device)

        # ----------------------------------------------------
        # Compute Mahalanobis distance
        # ----------------------------------------------------

        diff = feature - mean

        distance = torch.sqrt(
            torch.matmul(
                torch.matmul(diff.unsqueeze(0), inv_cov),
                diff.unsqueeze(1)
            )
        ).squeeze()

        return {

            "mahalanobis_distance": float(distance.item()),

            "mahalanobis_available": True

        }