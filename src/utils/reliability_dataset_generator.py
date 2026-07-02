"""
=========================================================
UAIRE - Universal AI Reliability Engine

Universal Reliability Dataset Generator (URDG)

Generates one reliability profile per sample.
=========================================================
"""

import pandas as pd


class ReliabilityDatasetGenerator:

    def __init__(self, extractors):

        self.extractors = extractors

        self.rows = []

    def process_sample(
        self,
        model,
        input_tensor,
        output,
        label,
        **kwargs
    ):

        sample = {}

        for extractor in self.extractors:

            signals = extractor.extract(
                model=model,
                input_tensor=input_tensor,
                output=output,
                **kwargs
            )

            sample.update(signals)

        prediction = output.argmax(dim=1).item()

        sample["prediction"] = prediction
        sample["true_label"] = label
        sample["failure_label"] = int(prediction != label)

        self.rows.append(sample)

    def to_dataframe(self):

        return pd.DataFrame(self.rows)

    def save_csv(self, path):

        df = self.to_dataframe()

        df.to_csv(path, index=False)

        print(f"Saved {len(df)} samples to {path}")