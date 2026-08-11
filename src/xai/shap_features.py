"""
=========================================================
UAIRE

SHAP Meta-Model Explainer

Explains the Random Forest meta-failure predictor without
retraining or modifying the research pipeline.
=========================================================
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import joblib
import numpy as np
import pandas as pd


class SHAPMetaExplainer:

    """
    SHAP wrapper for UAIRE's Random Forest meta-model.

    SHAP values are computed for class 1, which represents
    failure probability in the meta-failure predictor.
    """

    def __init__(self, background_rows=256):

        import shap

        from src.config.paths import (
            FEATURE_DIR,
            UAIRE_FEATURE_ORDER,
            UAIRE_META_MODEL,
        )

        self.shap = shap
        self.meta_model = joblib.load(UAIRE_META_MODEL)
        self.feature_order = joblib.load(UAIRE_FEATURE_ORDER)
        self.background = self._load_background(
            FEATURE_DIR,
            background_rows
        )

        self.explainer = shap.TreeExplainer(
            self.meta_model,
            data=self.background,
            model_output="probability",
        )

        self.background_explanation = self._class_one_explanation(
            self.explainer(self.background)
        )

    def _load_background(self, feature_dir, background_rows):

        candidates = [
            "reliability_dataset_v1.csv",
            "generalized_meta_dataset.csv",
            "resnet18_cifar10_features.csv",
            "mobilenetv2_cifar10_features.csv",
        ]

        best = None
        best_overlap = -1

        for filename in candidates:
            path = Path(feature_dir) / filename
            if not path.exists():
                continue

            frame = pd.read_csv(path)
            overlap = len(
                [
                    feature
                    for feature in self.feature_order
                    if feature in frame.columns
                ]
            )

            if overlap > best_overlap:
                best = frame
                best_overlap = overlap

        if best is None:
            best = pd.DataFrame(
                np.zeros((1, len(self.feature_order))),
                columns=self.feature_order,
            )

        aligned = self._align_features(best)
        aligned = aligned.sample(
            n=min(background_rows, len(aligned)),
            random_state=42,
        )

        return aligned.reset_index(drop=True)

    def _align_features(self, features):

        if isinstance(features, dict):
            features = pd.DataFrame([features])

        aligned = features.copy()

        for feature in self.feature_order:
            if feature not in aligned.columns:
                aligned[feature] = 0.0

        aligned = aligned[self.feature_order]
        aligned = aligned.apply(
            pd.to_numeric,
            errors="coerce",
        )
        aligned = aligned.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)

        return aligned

    def _class_one_explanation(self, explanation):

        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)

        if values.ndim == 3:
            values = values[:, :, 1]

        if base_values.ndim == 2:
            base_values = base_values[:, 1]

        return self.shap.Explanation(
            values=values,
            base_values=base_values,
            data=explanation.data,
            feature_names=self.feature_order,
        )

    def explain(self, features):

        aligned = self._align_features(features)
        probability = self.meta_model.predict_proba(aligned)[0, 1]
        local_explanation = self._class_one_explanation(
            self.explainer(aligned)
        )

        contributions = pd.DataFrame(
            {
                "feature": self.feature_order,
                "value": aligned.iloc[0].values,
                "shap_value": local_explanation.values[0],
            }
        )
        contributions["impact"] = contributions["shap_value"].abs()
        contributions["direction"] = np.where(
            contributions["shap_value"] >= 0,
            "Increases failure risk",
            "Supports reliability",
        )
        contributions = contributions.sort_values(
            "impact",
            ascending=False,
        ).reset_index(drop=True)

        return {
            "aligned_features": aligned,
            "failure_probability": float(probability),
            "base_failure_probability": float(local_explanation.base_values[0]),
            "local_explanation": local_explanation,
            "background_explanation": self.background_explanation,
            "top_features": contributions.head(10),
            "all_contributions": contributions,
            "summary": self.natural_language_summary(
                contributions.head(10),
                probability,
            ),
        }

    def natural_language_summary(self, top_features, failure_probability):

        risk_features = top_features[
            top_features["shap_value"] > 0
        ]["feature"].head(3).tolist()
        reliable_features = top_features[
            top_features["shap_value"] < 0
        ]["feature"].head(3).tolist()

        def pretty(feature):
            return feature.replace("_", " ")

        if failure_probability >= 0.5:
            primary = risk_features or top_features["feature"].head(3).tolist()
            joined = ", ".join(pretty(feature) for feature in primary)
            return (
                "The prediction was considered unreliable primarily because "
                f"{joined} pushed the meta-model toward failure risk."
            )

        primary = reliable_features or top_features["feature"].head(3).tolist()
        joined = ", ".join(pretty(feature) for feature in primary)
        return (
            "The prediction was considered reliable primarily because "
            f"{joined} reduced the meta-model's estimated failure risk."
        )
