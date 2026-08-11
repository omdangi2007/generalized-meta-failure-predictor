"""
=========================================================
UAIRE

Universal Reliability Pipeline

Runs every reliability module and returns one
combined feature vector.
=========================================================
"""

from src.signal_extractors.registry import get_signal_extractors
from src.ood.registry import get_ood_detectors


class UniversalReliabilityPipeline:

    def __init__(self):

        self.signal_extractors = get_signal_extractors()

        self.ood_detectors = get_ood_detectors()

    def extract(
        self,
        model,
        input_tensor,
        output,
        activations,
        gradients,
        image
    ):

        sample = {

            "model": model,

            "input_tensor": input_tensor,

            "output": output,

            "activations": activations,

            "gradients": gradients,

            "image": image

        }

        reliability_features = {}

        # --------------------------------------------------
        # Signal Extractors
        # --------------------------------------------------

        for extractor in self.signal_extractors:

            features = extractor.extract(**sample)

            reliability_features.update(features)

        # --------------------------------------------------
        # OOD Detectors
        # --------------------------------------------------

        for detector in self.ood_detectors:

            features = detector.detect(**sample)

            reliability_features.update(features)

        return reliability_features