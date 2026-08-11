"""The immutable, shared column contract for every orchestrated dataset."""

from __future__ import annotations

METADATA_COLUMNS = (
    "image_id",
    "dataset",
    "split",
    "model_name",
    "architecture",
    "checkpoint_name",
    "number_of_classes",
    "prediction",
    "true_label",
    "failure_label",
)

# This list mirrors the existing UniversalReliabilityPipeline registry order.
# It is deliberately explicit: a new extractor must update this contract rather
# than changing the column order accidentally at runtime.
UAIRE_FEATURE_COLUMNS = (
    "confidence", "entropy", "prediction_margin",
    "activation_mean", "activation_std", "activation_max", "activation_min",
    "activation_energy", "activation_sparsity", "dead_neuron_ratio",
    "positive_activation_ratio", "activation_entropy",
    "gradient_mean", "gradient_std", "gradient_max", "gradient_min",
    "gradient_norm", "gradient_energy", "gradient_sparsity", "gradient_entropy",
    "brightness", "contrast", "image_entropy", "blur_score",
    "laplacian_variance", "edge_density",
    "msp_probability", "msp_ood_score", "energy_score", "normalized_energy",
    "mahalanobis_distance", "mahalanobis_available",
)

STANDARDIZED_COLUMNS = METADATA_COLUMNS + UAIRE_FEATURE_COLUMNS
