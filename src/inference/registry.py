"""
=========================================================
UAIRE

Inference Registry
=========================================================
"""

from .predictor import UAIREPredictor


def get_predictor(model, device=None, target_layer=None, class_names=None):

    return UAIREPredictor(
        model=model,
        device=device,
        target_layer=target_layer,
        class_names=class_names,
    )
