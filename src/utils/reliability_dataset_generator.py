"""
=========================================================
UAIRE - Universal AI Reliability Engine

Universal Reliability Dataset Generator

Version 2

Uses the UAIRE Pipeline instead of manually calling
individual extractors.
=========================================================
"""

import pandas as pd


class ReliabilityDatasetGenerator:

    def __init__(self, pipeline):

        self.pipeline = pipeline
        self.rows = []

    def process_sample(
        self,
        model,
        input_tensor,
        output,
        activations,
        gradients,
        image,
        label
    ):

        # ----------------------------------------------------
        # Extract all reliability features
        # ----------------------------------------------------

        sample = self.pipeline.extract(

            model=model,

            input_tensor=input_tensor,

            output=output,

            activations=activations,

            gradients=gradients,

            image=image

        )

        # ----------------------------------------------------
        # Prediction Information
        # ----------------------------------------------------

        prediction = output.argmax(dim=1).item()

        sample["prediction"] = prediction
        sample["true_label"] = int(label)
        sample["failure_label"] = int(prediction != label)

        self.rows.append(sample)

    def to_dataframe(self):

        return pd.DataFrame(self.rows)

    def save_csv(self, path):

        df = self.to_dataframe()

        df.to_csv(path, index=False)

        print("=" * 60)
        print("Reliability Dataset Saved Successfully")
        print("=" * 60)
        print(f"Samples : {len(df)}")
        print(f"Features: {len(df.columns)}")
        print(f"Saved to: {path}")