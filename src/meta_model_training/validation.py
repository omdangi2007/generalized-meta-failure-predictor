"""Input validation that preserves the Phase 3 universal dataset unchanged."""

from __future__ import annotations

import math

import pandas as pd

from src.dataset_orchestration.schema import METADATA_COLUMNS, STANDARDIZED_COLUMNS, UAIRE_FEATURE_COLUMNS


def load_and_validate_dataset(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != list(STANDARDIZED_COLUMNS):
        raise ValueError("Universal Meta Dataset schema does not exactly match the standardized UAIRE schema.")
    if frame.empty:
        raise ValueError("Universal Meta Dataset has no rows.")
    if frame.isna().any().any():
        raise ValueError(f"Universal Meta Dataset contains null values: {frame.columns[frame.isna().any()].tolist()}")
    if frame.columns.duplicated().any():
        raise ValueError("Universal Meta Dataset contains duplicate columns.")
    for column in UAIRE_FEATURE_COLUMNS:
        values = frame[column]
        if column == "mahalanobis_available":
            continue
        if not values.map(lambda value: isinstance(value, (int, float)) and math.isfinite(float(value))).all():
            raise ValueError(f"Feature '{column}' has non-finite values.")
    if not all(frame[column].notna().all() for column in METADATA_COLUMNS):
        raise ValueError("Required metadata contains null values.")
    if not frame["failure_label"].isin([0, 1]).all():
        raise ValueError("failure_label must be binary.")
    expected_failure = (frame["prediction"].astype(int) != frame["true_label"].astype(int)).astype(int)
    if not expected_failure.equals(frame["failure_label"].astype(int)):
        raise ValueError("failure_label is inconsistent with prediction and true_label.")
    return frame
