"""
=========================================================
UAIRE

Predictor

Main entry point of the framework.
=========================================================
"""

import joblib

from .reliability_score import ReliabilityScore


class UAIREPredictor:

    def __init__(

        self,

        model_path="../models/uaire_meta_model.pkl",

        feature_path="../models/uaire_feature_order.pkl"

    ):

        self.meta_model = joblib.load(model_path)

        self.feature_order = joblib.load(feature_path)

    def predict(

        self,

        feature_vector,

        predicted_class,

        confidence,

        reasons

    ):

        feature_vector = feature_vector[self.feature_order]

        failure_probability = self.meta_model.predict_proba(

            feature_vector

        )[0, 1]

        reliability = ReliabilityScore.compute(

            failure_probability

        )

        decision = ReliabilityScore.decision(

            reliability

        )

        return {

            "predicted_class": predicted_class,

            "confidence": confidence,

            "failure_probability": failure_probability,

            "reliability_score": reliability,

            "decision": decision,

            "reasons": reasons

        }