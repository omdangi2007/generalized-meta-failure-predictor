"""Quality, statistical, and publication-figure outputs for a merged dataset."""

from __future__ import annotations

import os
from pathlib import Path

# Some deployments keep the user home read-only. Configure this before pyplot
# is imported so figure generation stays self-contained and repeatable.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/uaire-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset_orchestration.schema import METADATA_COLUMNS, UAIRE_FEATURE_COLUMNS


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def quality_report(frame: pd.DataFrame, source_reports: list[dict]) -> dict:
    metadata_consistent = all(frame[column].notna().all() for column in METADATA_COLUMNS)
    return {
        "total_rows": int(len(frame)),
        "rows_per_architecture": _counts(frame["architecture"]),
        "rows_per_dataset": _counts(frame["dataset"]),
        "feature_count": len(UAIRE_FEATURE_COLUMNS),
        "missing_values": {column: int(count) for column, count in frame.isna().sum().items() if count},
        "missing_value_count": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_columns": frame.columns[frame.columns.duplicated()].tolist(),
        "metadata_consistency": {
            "all_required_metadata_present": metadata_consistent,
            "invalid_source_runs": [report["id"] for report in source_reports if not report["validation_passed"]],
        },
        "failure_label_distribution": _counts(frame["failure_label"]),
        "class_distribution": _counts(frame["true_label"]),
    }


def dataset_statistics(frame: pd.DataFrame) -> dict:
    feature_frame = frame.loc[:, list(UAIRE_FEATURE_COLUMNS)].copy()
    feature_frame["mahalanobis_available"] = feature_frame["mahalanobis_available"].astype(float)
    describe = feature_frame.describe(percentiles=[0.25, 0.5, 0.75]).T
    feature_stats = {name: {stat: float(value) for stat, value in row.dropna().items()} for name, row in describe.iterrows()}
    return {
        "total_rows": int(len(frame)),
        "number_of_architectures": int(frame["architecture"].nunique()),
        "number_of_datasets": int(frame["dataset"].nunique()),
        "samples_per_architecture": _counts(frame["architecture"]),
        "samples_per_dataset": _counts(frame["dataset"]),
        "failure_rate_per_architecture": {str(key): float(value) for key, value in frame.groupby("architecture")["failure_label"].mean().sort_index().items()},
        "failure_rate_per_dataset": {str(key): float(value) for key, value in frame.groupby("dataset")["failure_label"].mean().sort_index().items()},
        "basic_feature_statistics": feature_stats,
    }


def write_plots(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def bar(column: str, title: str, filename: str) -> None:
        counts = frame[column].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        counts.plot.bar(ax=ax, color="#277DA1")
        ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("Samples")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=300); plt.close(fig)
        files.append(filename)

    bar("architecture", "Architecture Distribution", "architecture_distribution.png")
    bar("dataset", "Dataset Distribution", "dataset_distribution.png")
    bar("failure_label", "Failure-Label Distribution", "failure_label_distribution.png")

    numeric = frame.loc[:, list(UAIRE_FEATURE_COLUMNS)].copy()
    numeric["mahalanobis_available"] = numeric["mahalanobis_available"].astype(float)
    correlation = numeric.corr()
    fig, ax = plt.subplots(figsize=(15, 12))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(correlation))); ax.set_xticklabels(correlation.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(correlation))); ax.set_yticklabels(correlation.index, fontsize=6)
    ax.set_title("Reliability Feature Correlation")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout(); fig.savefig(output_dir / "reliability_feature_correlation_heatmap.png", dpi=300); plt.close(fig)
    files.append("reliability_feature_correlation_heatmap.png")

    missing = frame.isna().sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(len(missing)), missing.values, color="#F94144")
    ax.set_title("Feature Missingness"); ax.set_xlabel("Columns"); ax.set_ylabel("Missing values")
    ax.set_xticks(range(len(missing))); ax.set_xticklabels(missing.index, rotation=90, fontsize=6)
    fig.tight_layout(); fig.savefig(output_dir / "feature_missingness.png", dpi=300); plt.close(fig)
    files.append("feature_missingness.png")
    return files
