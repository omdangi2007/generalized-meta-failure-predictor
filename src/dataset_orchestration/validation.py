"""Strict validation of the standardized output contract."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd

from .schema import METADATA_COLUMNS, STANDARDIZED_COLUMNS, UAIRE_FEATURE_COLUMNS


def validate_dataframe(frame: pd.DataFrame, config) -> dict:
    errors: list[str] = []
    columns = list(frame.columns)
    duplicates = [name for name, count in Counter(columns).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate columns: {duplicates}")
    if columns != list(STANDARDIZED_COLUMNS):
        errors.append("columns do not exactly match the standardized schema and order")
    missing = [name for name in STANDARDIZED_COLUMNS if name not in frame.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if frame.empty:
        errors.append("dataset contains no rows")
    if frame.isna().any().any():
        errors.append(f"NaN/null values in columns: {frame.columns[frame.isna().any()].tolist()}")
    numeric_columns = [name for name in UAIRE_FEATURE_COLUMNS if name != "mahalanobis_available"]
    for name in numeric_columns:
        if name in frame and not frame[name].map(lambda value: isinstance(value, (int, float)) and math.isfinite(float(value))).all():
            errors.append(f"non-finite or non-numeric values in '{name}'")
    expected_metadata = (
        ("dataset", config.dataset),
        ("split", config.split),
        ("model_name", config.model_name or config.backbone),
        ("architecture", config.backbone),
        ("checkpoint_name", Path(config.checkpoint).name),
        ("number_of_classes", frame["number_of_classes"].iloc[0] if "number_of_classes" in frame and not frame.empty else None),
    )
    for name, expected in expected_metadata:
        if name in frame and set(frame[name].astype(str)) != {str(expected)}:
            errors.append(f"inconsistent '{name}' metadata")
    if "number_of_classes" in frame and (frame["number_of_classes"].astype(int) < 2).any():
        errors.append("number_of_classes must be at least 2")
    if {"prediction", "true_label", "number_of_classes"}.issubset(frame.columns):
        valid_labels = (frame["prediction"].astype(int) >= 0) & (frame["prediction"].astype(int) < frame["number_of_classes"].astype(int))
        valid_labels &= (frame["true_label"].astype(int) >= 0) & (frame["true_label"].astype(int) < frame["number_of_classes"].astype(int))
        if not valid_labels.all():
            errors.append("prediction or true_label is outside number_of_classes")
    if "image_id" in frame and frame["image_id"].duplicated().any():
        errors.append("image_id values are not unique")
    if "failure_label" in frame and not frame["failure_label"].isin([0, 1]).all():
        errors.append("failure_label must be 0 or 1")
    if {"prediction", "true_label", "failure_label"}.issubset(frame.columns):
        expected = (frame["prediction"].astype(int) != frame["true_label"].astype(int)).astype(int)
        if not expected.equals(frame["failure_label"].astype(int)):
            errors.append("failure_label does not equal prediction != true_label")
    return {
        "passed": not errors,
        "row_count": int(len(frame)),
        "column_count": int(len(columns)),
        "schema": list(STANDARDIZED_COLUMNS),
        "checks": {
            "no_missing_columns": not missing,
            "no_duplicate_columns": not duplicates,
            "correct_feature_order": columns == list(STANDARDIZED_COLUMNS),
            "no_nan_values": not frame.isna().any().any(),
            "consistent_metadata": not any("metadata" in error for error in errors),
        },
        "errors": errors,
    }
