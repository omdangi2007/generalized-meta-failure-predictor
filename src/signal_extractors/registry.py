"""
=========================================================
UAIRE

Signal Extractor Registry
=========================================================
"""

from .prediction_confidence import PredictionConfidenceExtractor
from .activation_signals import ActivationSignalExtractor
from .gradient_signals import GradientSignalExtractor
from .input_quality import InputQualityExtractor


def get_signal_extractors():

    return [

        PredictionConfidenceExtractor(),

        ActivationSignalExtractor(),

        GradientSignalExtractor(),

        InputQualityExtractor()

    ]