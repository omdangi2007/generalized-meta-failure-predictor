"""
=========================================================
UAIRE - Universal AI Reliability Engine

Base class for all reliability signal extractors.

Every reliability extractor must inherit from this class
and implement the extract() method.

Author: Om Dangi
=========================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ReliabilitySignalExtractor(ABC):
    """
    Abstract base class for all reliability signal extractors.

    Every subclass must implement the extract() method.
    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Name of the reliability extractor.
        """
        self.name = name

    @abstractmethod
    def extract(
        self,
        model,
        input_tensor,
        output,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract reliability signals.

        Parameters
        ----------
        model : torch.nn.Module
            Trained neural network.

        input_tensor : torch.Tensor
            Input image tensor.

        output : torch.Tensor
            Raw model output (logits).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing reliability signals.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"