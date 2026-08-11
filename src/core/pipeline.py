"""
=========================================================
UAIRE

Universal AI Reliability Pipeline

This is the heart of UAIRE.

Responsibilities:
1. Run all Signal Extractors
2. Run all OOD Detectors
3. Merge everything into one feature dictionary

Author: Om V. Dangi
=========================================================
"""

from src.signal_extractors.registry import get_signal_extractors
from src.ood.registry import get_ood_detectors


class UAIREPipeline:

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

        features = {}

        # -------------------------------------------------
        # Signal Extractors
        # -------------------------------------------------

        for extractor in self.signal_extractors:

            result = extractor.extract(**sample)

            features.update(result)

        # -------------------------------------------------
        # OOD Detectors
        # -------------------------------------------------

        for detector in self.ood_detectors:

            result = detector.detect(**sample)

            features.update(result)

        return features

    def feature_names(self):

        names = []

        dummy = {}

        for extractor in self.signal_extractors:

            names.extend(
                extractor.extract(**dummy).keys()
            )

        for detector in self.ood_detectors:

            names.extend(
                detector.detect(**dummy).keys()
            )

        return names