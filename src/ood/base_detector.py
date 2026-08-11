"""
=========================================================
UAIRE

Base Out-of-Distribution Detector
=========================================================
"""

from abc import ABC, abstractmethod


class OODDetector(ABC):

    def __init__(self, name):

        self.name = name

    @abstractmethod
    def detect(
        self,
        model,
        input_tensor,
        output,
        activations=None,
        gradients=None,
        **kwargs
    ):
        """
        Returns a dictionary of OOD-related features.
        """
        pass