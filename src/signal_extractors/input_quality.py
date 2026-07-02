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

        # ---------------------------------------------------
        # Convert RGB to Gray if needed
        # ---------------------------------------------------

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # ---------------------------------------------------
        # Ensure uint8 image
        # ---------------------------------------------------

        if gray.dtype != np.uint8:

            # If image is normalized (0-1)
            if gray.max() <= 1.0:
                gray = gray * 255.0

            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # ---------------------------------------------------
        # Debug Information
        # ---------------------------------------------------

        print("\n========== INPUT QUALITY DEBUG ==========")
        print(f"Shape      : {gray.shape}")
        print(f"Dtype      : {gray.dtype}")
        print(f"Min Pixel  : {gray.min()}")
        print(f"Max Pixel  : {gray.max()}")

        # ---------------------------------------------------
        # Brightness
        # ---------------------------------------------------

        brightness = np.mean(gray)

        # ---------------------------------------------------
        # Contrast
        # ---------------------------------------------------

        contrast = np.std(gray)

        # ---------------------------------------------------
        # Image Entropy
        # ---------------------------------------------------

        histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        histogram = histogram / histogram.sum()

        image_entropy = -np.sum(
            histogram * np.log2(histogram + 1e-12)
        )

        # ---------------------------------------------------
        # Laplacian Variance (Blur)
        # ---------------------------------------------------

        try:

            laplacian = cv2.Laplacian(
                gray,
                cv2.CV_32F
            )

            laplacian_variance = float(laplacian.var())

        except Exception as e:

            print("Laplacian Error:", e)

            laplacian_variance = 0.0

        blur_score = laplacian_variance

        # ---------------------------------------------------
        # Edge Density
        # ---------------------------------------------------

        edges = cv2.Canny(gray, 100, 200)

        edge_density = float(np.mean(edges > 0))

        # ---------------------------------------------------
        # Return Signals
        # ---------------------------------------------------

        return {

            "brightness": float(brightness),

            "contrast": float(contrast),

            "image_entropy": float(image_entropy),

            "blur_score": float(blur_score),

            "laplacian_variance": float(laplacian_variance),

            "edge_density": float(edge_density)

        }