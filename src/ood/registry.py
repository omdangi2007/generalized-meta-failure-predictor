"""
=========================================================
UAIRE

OOD Detector Registry
=========================================================
"""

from .msp_detector import MSPDetector
from .energy_detector import EnergyDetector
from .mahalanobis_detector import MahalanobisDetector


def get_ood_detectors():

    return [

        MSPDetector(),

        EnergyDetector(),

        MahalanobisDetector()

    ]