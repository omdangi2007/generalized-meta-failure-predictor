"""Evaluation, explainability, and Phase 4 review artifact generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/uaire-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)


def _json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate(model, features: pd.DataFrame, target: pd.Series, plot_dir: Path) -> tuple[dict, dict]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "sample_count": int(len(target)),
    }
    report = classification_report(target, predictions, output_dict=True, zero_division=0)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4)); ConfusionMatrixDisplay.from_predictions(target, predictions, ax=ax, colorbar=False)
    ax.set_title("Test Confusion Matrix"); fig.tight_layout(); fig.savefig(plot_dir / "confusion_matrix.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); RocCurveDisplay.from_predictions(target, probabilities, ax=ax)
    ax.set_title("Test ROC Curve"); fig.tight_layout(); fig.savefig(plot_dir / "roc_curve.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); PrecisionRecallDisplay.from_predictions(target, probabilities, ax=ax)
    ax.set_title("Test Precision-Recall Curve"); fig.tight_layout(); fig.savefig(plot_dir / "pr_curve.png", dpi=300); plt.close(fig)
    fraction_positive, mean_predicted = calibration_curve(target, probabilities, n_bins=min(10, len(target)))
    fig, ax = plt.subplots(figsize=(5, 4)); ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_positive, marker="o", label="Random Forest"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction positive")
    ax.set_title("Test Calibration Curve"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "calibration_curve.png", dpi=300); plt.close(fig)
    return metrics, report


def feature_importance(model, feature_names: list[str], output_dir: Path) -> tuple[pd.DataFrame, str]:
    table = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    table.to_csv(output_dir / "feature_importance.csv", index=False)
    top = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7)); ax.barh(top["feature"], top["importance"], color="#277DA1")
    ax.set_title("Top 20 Random Forest Feature Importances"); ax.set_xlabel("Importance"); fig.tight_layout()
    fig.savefig(output_dir / "plots" / "feature_importance_top20.png", dpi=300); plt.close(fig)
    top10 = ", ".join(f"{row.feature} ({row.importance:.3f})" for row in table.head(10).itertuples())
    explanation = (
        "The generalized meta-model primarily uses the following reliability signals: " + top10 + ". "
        "These importances describe associations within the merged source dataset; they do not establish causal effects."
    )
    (output_dir / "explainability_summary.md").write_text(explanation + "\n", encoding="utf-8")
    return table, explanation


def shap_plots(model, features: pd.DataFrame, output_dir: Path, sample_size: int, seed: int) -> dict:
    """Create SHAP artifacts without changing model behavior; failures are reported."""
    try:
        import shap
        sample = features.sample(n=min(sample_size, len(features)), random_state=seed)
        explanation = shap.TreeExplainer(model)(sample)
        values = explanation.values
        if values.ndim == 3:
            values = values[:, :, 1]
        plt.figure(); shap.summary_plot(values, sample, show=False, max_display=20)
        plt.tight_layout(); plt.savefig(output_dir / "plots" / "shap_summary.png", dpi=300, bbox_inches="tight"); plt.close()
        plt.figure(); shap.summary_plot(values, sample, plot_type="bar", show=False, max_display=20)
        plt.tight_layout(); plt.savefig(output_dir / "plots" / "shap_bar.png", dpi=300, bbox_inches="tight"); plt.close()
        return {"available": True, "sample_count": int(len(sample))}
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def write_phase_review(path: Path, statistics: dict, evaluation: dict, top_features: pd.DataFrame, hyperparameters: dict, shap_status: dict) -> None:
    top20 = "\n".join(f"{index + 1}. {row.feature}: {row.importance:.6f}" for index, row in enumerate(top_features.head(20).itertuples()))
    text = f"""# Phase 4 Review Report

## 1. Research Context

- Research goal: learn generalized reliability patterns across supported visual backbones and datasets.
- Research question: can unchanged UAIRE reliability signals predict classification failure in a shared research dataset?
- Engineering deliverable: reproducible Random Forest training artifacts, evaluations, and explainability outputs.
- Research deliverable: one validated generalized failure-prediction model.

## 2. Architecture Summary

`universal_meta_dataset.csv → schema/metadata validation → stratified train/validation/test splits → Random Forest → evaluation + explainability → persisted artifacts`.

This preserves the standardized feature contract and avoids altering backbone, adapter, extractor, predictor, or UI behavior.

## 3. Files Modified

No pre-existing application files were modified by Phase 4.

## 4. New Files Created

The `src/meta_model_training` package provides configuration, immutable-dataset validation, training, reporting, and a CLI. Result files are listed in Section 10.

## 5. Directory Structure Changes

`results/meta_model/` contains model artifacts, JSON reports, split metadata, feature importance, explainability narrative, and `plots/`.

## 6. Implementation Summary

- Completed: schema validation, stratified splitting, Random Forest training, persistence, evaluation, feature importance, SHAP plots, reports.
- Partially implemented: SHAP is environment-dependent; its status is recorded below rather than silently omitted.
- Not implemented: cross-architecture testing, UI integration, and new model training beyond the required meta-model.

## 7. Training Summary

- Dataset: Universal Meta Dataset
- Samples: {statistics['sample_count']}
- Architectures: {statistics['architecture_count']}
- Datasets: {statistics['dataset_count']}
- Failure rate: {statistics['failure_rate']:.4f}
- Target distribution: {statistics['target_distribution']}
- Training time: {statistics['training_time_seconds']:.2f} seconds
- Hyperparameters: {hyperparameters}

## 8. Evaluation Results

- Accuracy: {evaluation['test']['accuracy']:.4f}
- Precision: {evaluation['test']['precision']:.4f}
- Recall: {evaluation['test']['recall']:.4f}
- F1: {evaluation['test']['f1']:.4f}
- ROC-AUC: {evaluation['test']['roc_auc']:.4f}
- Confusion matrix, ROC, PR, and calibration figures: `plots/`.
- Classification report: `classification_report.json`.

## 9. Explainability Results

Top 20 features:

{top20}

SHAP status: {shap_status}. Feature importances are predictive associations, not causal claims.

## 10. Example Outputs

`model.pkl`, `feature_order.pkl`, `training_configuration.yaml`, `model_metadata.json`, `training_statistics.json`, `split_metadata.json`, `classification_report.json`, `feature_importance.csv`, `explainability_summary.md`, and `plots/`.

## 11. Validation Results

The input schema/order, metadata, target consistency, feature finiteness, stratified split viability, and persisted model prediction path were validated.

## 12. Known Limitations

Results depend on available Phase 3 source coverage and class balance. Random Forest probabilities may need calibration review. SHAP availability depends on the installed runtime. This phase does not test architecture-held-out generalization.

## 13. Research Contribution

This phase provides a repeatable empirical test of whether a common set of reliability signals carries failure-prediction value across merged model and dataset sources.

## 14. Patent Contribution

Yes. The generalized model and its reproducible feature contract strengthen evidence for a model-agnostic reliability-estimation workflow, while not by themselves establishing patentability.

## 15. Next Recommended Phase

Cross-architecture held-out evaluation.
"""
    path.write_text(text, encoding="utf-8")
