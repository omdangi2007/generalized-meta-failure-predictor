"""
=========================================================
UAIRE

OOD Detector Registry
=========================================================
"""

from .msp_detector import MSPDetector


def get_ood_detectors():

    return [

        MSPDetector()

    ]