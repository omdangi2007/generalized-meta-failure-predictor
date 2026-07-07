"""
=========================================================
UAIRE

Automatic Reason Generator

Generates human-readable explanations from
the extracted reliability features.
=========================================================
"""


class ReasonGenerator:

    @staticmethod
    def generate(features):

        reasons = []

        confidence = features.get("confidence", 0.0)
        entropy = features.get("entropy", 0.0)
        margin = features.get("prediction_margin", 0.0)

        mahalanobis = features.get(
            "mahalanobis_distance",
            None
        )

        energy = features.get(
            "normalized_energy",
            None
        )

        # --------------------------------------------
        # Confidence
        # --------------------------------------------

        if confidence >= 0.90:
            reasons.append("High confidence prediction")

        elif confidence <= 0.50:
            reasons.append("Low confidence prediction")

        # --------------------------------------------
        # Entropy
        # --------------------------------------------

        if entropy <= 0.30:
            reasons.append("Low predictive uncertainty")

        elif entropy >= 1.0:
            reasons.append("High predictive uncertainty")

        # --------------------------------------------
        # Margin
        # --------------------------------------------

        if margin >= 0.50:
            reasons.append("Large prediction margin")

        # --------------------------------------------
        # Energy
        # --------------------------------------------

        if energy is not None:

            if energy >= 0.90:
                reasons.append("Normal energy score")

            else:
                reasons.append("Unusual energy score")

        # --------------------------------------------
        # Mahalanobis
        # --------------------------------------------

        if mahalanobis is not None:

            if mahalanobis <= 25:
                reasons.append("Sample appears in-distribution")

            else:
                reasons.append("Sample is far from training distribution")

        if len(reasons) == 0:
            reasons.append("No dominant reliability indicators detected.")

        return reasons