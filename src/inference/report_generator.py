"""
=========================================================
UAIRE

Reliability Report Generator
=========================================================
"""


class ReportGenerator:

    @staticmethod
    def generate(result):

        print("\n" + "=" * 60)
        print("UAIRE RELIABILITY REPORT")
        print("=" * 60)

        print()

        print(f"Predicted Class      : {result['predicted_class']}")
        print(f"Confidence           : {result['confidence']:.4f}")

        print()

        print(f"Failure Probability  : {result['failure_probability']:.4f}")

        print(f"Reliability Score    : {result['reliability_score']:.2f}%")

        print(f"Decision             : {result['decision']}")

        print()

        print("Reasoning:")

        for reason in result["reasons"]:
            print(f" • {reason}")

        print("=" * 60)