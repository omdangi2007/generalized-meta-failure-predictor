"""
=========================================================
UAIRE

Meta Predictor

Loads the trained meta-model and predicts
the probability that the base model prediction
will fail.
=========================================================
"""

import joblib
import pandas as pd


class MetaPredictor:

    def __init__(
        self,
        model_path="../models/uaire_meta_model.pkl",
        feature_order_path="../models/uaire_feature_order.pkl"
    ):

        self.meta_model = joblib.load(model_path)

        self.feature_order = joblib.load(
            feature_order_path
        )

    def predict(self, features):

        if isinstance(features, dict):

            features = pd.DataFrame([features])

        features = features[self.feature_order]

        probability = self.meta_model.predict_proba(
            features
        )[0, 1]

        return float(probability)