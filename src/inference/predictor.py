"""
=========================================================
UAIRE

Universal AI Reliability Engine

Predictor

Main entry point of the framework.

Author:
Om V. Dangi

=========================================================
"""

import numpy as np
import pandas as pd
import torch

from PIL import Image

from src.models.model_loader import (
    load_resnet18_cifar10
)

from src.utils.image_preprocessor import (
    ImagePreprocessor
)

from src.utils.neural_state_collector import (
    NeuralStateCollector
)

from src.core.pipeline import (
    UAIREPipeline
)

from src.inference.meta_predictor import (
    MetaPredictor
)

from src.inference.reason_generator import (
    ReasonGenerator
)

from src.inference.reliability_score import (
    ReliabilityScore
)

from src.inference.report_generator import (
    ReportGenerator
)


class UAIREPredictor:

    """
    =====================================================
    Universal Predictor

    Complete End-to-End Reliability Pipeline

    Image
        ↓
    Preprocessing
        ↓
    Backbone Model
        ↓
    Neural State Collector
        ↓
    UAIRE Pipeline
        ↓
    Meta Predictor
        ↓
    Reliability Score
        ↓
    Report
    =====================================================
    """

    def __init__(

        self,

        device=None,

        model=None,

        class_names=None

    ):

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        if device is None:

            device = (

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        self.device = device

        # -------------------------------------------------
        # Load Backbone Model
        # -------------------------------------------------

        if model is None:

            self.model = load_resnet18_cifar10(

                device=device

            )

        else:

            self.model = model

        self.model.eval()

        # -------------------------------------------------
        # Target Layer
        # -------------------------------------------------

        self.target_layer = self.model.layer4[-1]

        # -------------------------------------------------
        # Image Preprocessor
        # -------------------------------------------------

        self.preprocessor = ImagePreprocessor()

        # -------------------------------------------------
        # Neural State Collector
        # -------------------------------------------------

        self.collector = NeuralStateCollector(

            self.target_layer

        )

        # Register Hooks

        self.collector.register_hooks()

        # -------------------------------------------------
        # UAIRE Pipeline
        # -------------------------------------------------

        self.pipeline = UAIREPipeline()

        # -------------------------------------------------
        # Meta Predictor
        # -------------------------------------------------

        self.meta_predictor = MetaPredictor()

        # -------------------------------------------------
        # CIFAR10 Labels
        # -------------------------------------------------

        if class_names is None:

            self.class_names = [

                "Airplane",

                "Automobile",

                "Bird",

                "Cat",

                "Deer",

                "Dog",

                "Frog",

                "Horse",

                "Ship",

                "Truck"

            ]

        else:

            self.class_names = class_names

    # =====================================================
    # Load Image
    # =====================================================

    def _load_image(

        self,

        image

    ):

        if isinstance(

            image,

            str

        ):

            image = Image.open(

                image

            ).convert("RGB")

        return image

    # =====================================================
    # Preprocess Image
    # =====================================================

    def _prepare_input(

        self,

        image

    ):

        tensor = self.preprocessor.process(

            image

        )

        tensor = tensor.to(

            self.device

        )

        return tensor
    # =====================================================
    # Collect Neural State
    # =====================================================

    def _collect_states(self, input_tensor):

        states = self.collector.collect(

            model=self.model,

            image=input_tensor,

            compute_gradients=True

        )

        return states

    # =====================================================
    # Extract Reliability Features
    # =====================================================

    def _extract_features(

        self,

        input_tensor,

        states,

        original_image

    ):

        features = self.pipeline.extract(

            model=self.model,

            input_tensor=input_tensor,

            output=states["logits"],

            activations=states["activations"],

            gradients=states["gradients"],

            image=original_image

        )

        return features

    # =====================================================
    # Convert Features -> DataFrame
    # =====================================================

    def _build_feature_dataframe(

        self,

        features

    ):

        df = pd.DataFrame(

            [features]

        )

        return df

    # =====================================================
    # Prediction Information
    # =====================================================

    def _prediction_information(

        self,

        logits

    ):

        probabilities = torch.softmax(

            logits,

            dim=1

        )

        prediction = torch.argmax(

            probabilities,

            dim=1

        ).item()

        confidence = probabilities[0][prediction].item()

        return {

            "class_index": prediction,

            "class_name": self.class_names[prediction],

            "confidence": confidence,

            "probabilities": probabilities

        }

    # =====================================================
    # Reliability Prediction
    # =====================================================

    def _predict_failure(

        self,

        feature_dataframe

    ):

        failure_probability = self.meta_predictor.predict(

            feature_dataframe

        )

        reliability_score = ReliabilityScore.compute(

            failure_probability

        )

        decision = ReliabilityScore.decision(

            reliability_score

        )

        return {

            "failure_probability": failure_probability,

            "reliability_score": reliability_score,

            "decision": decision

        }
    # =====================================================
    # Predict
    # =====================================================

    def predict(self, image):

        # -------------------------------------------------
        # Load Original Image
        # -------------------------------------------------

        image = self._load_image(image)

        # -------------------------------------------------
        # Preserve Original Image for IQAM
        # -------------------------------------------------

        original_image = np.array(image)

        # -------------------------------------------------
        # Prepare Network Input
        # -------------------------------------------------

        input_tensor = self._prepare_input(image)

        # -------------------------------------------------
        # Collect Neural State
        # -------------------------------------------------

        states = self._collect_states(input_tensor)

        # -------------------------------------------------
        # Prediction Information
        # -------------------------------------------------

        prediction = self._prediction_information(
            states["logits"]
        )

        # -------------------------------------------------
        # Extract Reliability Features
        # -------------------------------------------------

        features = self._extract_features(

            input_tensor=input_tensor,

            states=states,

            original_image=original_image

        )

        # -------------------------------------------------
        # Convert Feature Dictionary → DataFrame
        # -------------------------------------------------

        feature_dataframe = self._build_feature_dataframe(
            features
        )

        # -------------------------------------------------
        # Failure Prediction
        # -------------------------------------------------

        reliability = self._predict_failure(
            feature_dataframe
        )

        # -------------------------------------------------
        # Automatic Explanation
        # -------------------------------------------------

        reasons = ReasonGenerator.generate(
            features
        )

        # -------------------------------------------------
        # Final Result Dictionary
        # -------------------------------------------------

        result = {

            "prediction": {

                "class": prediction["class_name"],

                "class_index": prediction["class_index"],

                "confidence": prediction["confidence"]

            },

            "reliability": {

                "failure_probability":
                    reliability["failure_probability"],

                "score":
                    reliability["reliability_score"],

                "decision":
                    reliability["decision"]

            },

            "ood": {

                "msp":
                    features.get(
                        "msp_probability",
                        None
                    ),

                "energy":
                    features.get(
                        "normalized_energy",
                        None
                    ),

                "mahalanobis":
                    features.get(
                        "mahalanobis_distance",
                        None
                    )

            },

            "signals": features,

            "reasons": reasons

        }

        return result
    # =====================================================
    # Cleanup
    # =====================================================

    def __del__(self):

        try:

            self.collector.remove_hooks()

        except Exception:

            pass