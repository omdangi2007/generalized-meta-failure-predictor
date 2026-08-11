"""
=========================================================
UAIRE

Reliability Score

Converts the meta-model failure probability into an
easy-to-understand reliability score.
=========================================================
"""


class ReliabilityScore:

    @staticmethod
    def compute(failure_probability):
        """
        Reliability = 100 - Failure Probability
        """

        reliability = (1.0 - failure_probability) * 100

        return round(reliability, 2)

    @staticmethod
    def decision(reliability_score):
        """
        Converts reliability score into a human-readable decision.
        """

        if reliability_score >= 90:
            return "Highly Reliable"

        elif reliability_score >= 75:
            return "Reliable"

        elif reliability_score >= 50:
            return "Use With Caution"

        else:
            return "Unreliable"