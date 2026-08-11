"""
=========================================================
UAIRE

Compute Mahalanobis Statistics

This utility extracts deep feature vectors from the
training dataset.

Later we will compute:

1. Mean Vector
2. Covariance Matrix
3. Inverse Covariance Matrix

Author: UAIRE
=========================================================
"""

import numpy as np
import torch

from tqdm import tqdm


def collect_features(
    model,
    collector,
    dataloader,
    device
):
    """
    Extract deep feature vectors from the training dataset.

    Returns
    -------
    np.ndarray
        Shape: (N, Feature_Dimension)
    """

    model.eval()

    features = []

    collector.register_hooks()

    try:

        with torch.no_grad():

            for images, _ in tqdm(
                dataloader,
                desc="Extracting Features"
            ):

                images = images.to(device)

                states = collector.collect(
                model,
                images,
                compute_gradients=False
            )

                if "activations" not in states:
                    raise RuntimeError(
                        "Collector did not return activations."
                    )

                activation = states["activations"]

                activation = activation.view(
                    activation.size(0),
                    -1
                )

                features.append(
                    activation.cpu().numpy()
                )

    finally:

        collector.remove_hooks()

    features = np.concatenate(
        features,
        axis=0
    )

    print("\nFeature Extraction Complete")
    print("Samples :", features.shape[0])
    print("Feature Dimension :", features.shape[1])

    return features

# ==========================================================
# Compute Mean Vector
# ==========================================================

def compute_mean(features):

    return np.mean(
        features,
        axis=0
    )


# ==========================================================
# Compute Covariance Matrix
# ==========================================================

def compute_covariance(features):

    covariance = np.cov(
        features,
        rowvar=False
    )

    return covariance


# ==========================================================
# Compute Inverse Covariance Matrix
# ==========================================================

def compute_inverse_covariance(
    covariance,
    regularization=1e-5
):

    covariance = covariance + (
        np.eye(covariance.shape[0]) * regularization
    )

    inverse_covariance = np.linalg.inv(
        covariance
    )

    return inverse_covariance