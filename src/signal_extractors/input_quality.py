"""
=========================================================
UAIRE - Universal AI Reliability Engine

Input Quality Assessment Module (IQAM)

Extracts:
1. Brightness
2. Contrast
3. Image Entropy
4. Blur Score
5. Laplacian Variance
6. Edge Density
=========================================================
"""

import cv2
import numpy as np

from .base_extractor import ReliabilitySignalExtractor


class InputQualityExtractor(ReliabilitySignalExtractor):

    def __init__(self):
        super().__init__("Input Quality Extractor")

    def extract(
        self,
        model,
        input_tensor,
        output,
        image=None,
        **kwargs
    ):

        if image is None:
            raise ValueError("Original image required.")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        brightness = np.mean(gray)

        contrast = np.std(gray)

        histogram = cv2.calcHist([gray],[0],None,[256],[0,256])
        histogram = histogram / histogram.sum()

        image_entropy = -np.sum(
            histogram * np.log2(histogram + 1e-12)
        )

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        laplacian_variance = laplacian.var()

        blur_score = laplacian_variance

        edges = cv2.Canny(gray,100,200)

        edge_density = np.mean(edges > 0)

        return {

            "brightness": float(brightness),

            "contrast": float(contrast),

            "image_entropy": float(image_entropy),

            "blur_score": float(blur_score),

            "laplacian_variance": float(laplacian_variance),

            "edge_density": float(edge_density)
        }