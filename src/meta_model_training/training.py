"""One generalized Random Forest, trained only from the immutable Phase 3 CSV."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.dataset_orchestration.metadata import sha256_file, write_json
from src.dataset_orchestration.schema import METADATA_COLUMNS, UAIRE_FEATURE_COLUMNS

from .config import MetaModelTrainingConfig
from .reporting import evaluate, feature_importance, shap_plots, write_phase_review
from .validation import load_and_validate_dataset


class UniversalMetaModelTrainer:
    def __init__(self, config: MetaModelTrainingConfig):
        self.config = config

    def run(self) -> Path:
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Phase 3 Universal Meta Dataset not found: {dataset_path}. Generate it before Phase 4 training.")
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        frame = load_and_validate_dataset(str(dataset_path))
        features = frame.loc[:, list(UAIRE_FEATURE_COLUMNS)].copy()
        features["mahalanobis_available"] = features["mahalanobis_available"].astype(int)
        target = frame["failure_label"].astype(int)
        self._validate_split_viability(target)
        indices = frame.index
        train_idx, held_out_idx = train_test_split(indices, test_size=self.config.validation_size + self.config.test_size, stratify=target, random_state=self.config.seed)
        validation_share = self.config.validation_size / (self.config.validation_size + self.config.test_size)
        validation_idx, test_idx = train_test_split(held_out_idx, test_size=1 - validation_share, stratify=target.loc[held_out_idx], random_state=self.config.seed)
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators, max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf, max_features=self.config.max_features,
            class_weight=self.config.class_weight, n_jobs=self.config.n_jobs, random_state=self.config.seed,
        )
        started = time.perf_counter(); model.fit(features.loc[train_idx], target.loc[train_idx]); elapsed = time.perf_counter() - started
        plot_dir = output / "plots"
        validation_metrics, _ = evaluate(model, features.loc[validation_idx], target.loc[validation_idx], plot_dir / "validation")
        test_metrics, classification = evaluate(model, features.loc[test_idx], target.loc[test_idx], plot_dir)
        importance, _ = feature_importance(model, list(UAIRE_FEATURE_COLUMNS), output)
        shap_status = shap_plots(model, features.loc[test_idx], output, self.config.shap_sample_size, self.config.seed)
        joblib.dump(model, output / "model.pkl")
        # Verify the persisted artifact is the model that was evaluated before
        # reporting the run as complete.
        reloaded_model = joblib.load(output / "model.pkl")
        persistence_valid = (reloaded_model.predict(features.loc[test_idx]) == model.predict(features.loc[test_idx])).all()
        if not persistence_valid:
            raise RuntimeError("Persisted model predictions differ from the evaluated model.")
        joblib.dump(list(UAIRE_FEATURE_COLUMNS), output / "feature_order.pkl")
        with (output / "training_configuration.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config.as_dict(), handle, sort_keys=False)
        split_metadata = self._split_metadata(frame, train_idx, validation_idx, test_idx)
        write_json(output / "split_metadata.json", split_metadata)
        write_json(output / "classification_report.json", classification)
        statistics = {
            "dataset_path": str(dataset_path.resolve()), "dataset_sha256": sha256_file(dataset_path),
            "sample_count": len(frame), "architecture_count": int(frame["architecture"].nunique()),
            "dataset_count": int(frame["dataset"].nunique()), "failure_rate": float(target.mean()),
            "target_distribution": {str(key): int(value) for key, value in Counter(target).items()},
            "feature_count": len(UAIRE_FEATURE_COLUMNS), "training_time_seconds": elapsed,
            "validation": validation_metrics, "test": test_metrics, "shap": shap_status,
            "model_persistence_validation": bool(persistence_valid),
        }
        write_json(output / "training_statistics.json", statistics)
        write_json(output / "model_metadata.json", {
            "algorithm": "RandomForestClassifier", "target": "failure_label", "metadata_columns": list(METADATA_COLUMNS),
            "feature_columns": list(UAIRE_FEATURE_COLUMNS), "scaler_required": False,
            "hyperparameters": model.get_params(), "training_dataset_sha256": statistics["dataset_sha256"],
        })
        write_phase_review(output / "PHASE_4_REVIEW_REPORT.md", statistics, {"test": test_metrics}, importance, model.get_params(), shap_status)
        return output

    def _validate_split_viability(self, target: pd.Series) -> None:
        counts = target.value_counts()
        if set(counts.index) != {0, 1}:
            raise ValueError("failure_label must contain both classes for generalized meta-model training.")
        if counts.min() < 5:
            raise ValueError("Each failure_label class needs at least five rows for stratified train/validation/test splits.")

    def _split_metadata(self, frame, train_idx, validation_idx, test_idx) -> dict:
        result = {"seed": self.config.seed, "stratified": True, "splits": {}}
        for name, indices in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
            subset = frame.loc[indices]
            result["splits"][name] = {
                "row_count": len(subset), "row_indices": [int(value) for value in indices],
                "failure_label_distribution": {str(key): int(value) for key, value in subset["failure_label"].value_counts().items()},
            }
        return result
