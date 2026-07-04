"""
=========================================================
UAIRE

Inference Registry
=========================================================
"""

from .predictor import UAIREPredictor


def get_predictor():

    return UAIREPredictor()