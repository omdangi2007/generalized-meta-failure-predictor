"""Orchestration only: load, extract unchanged UAIRE features, write, validate."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .config import OrchestratorConfig, save_config
from .feature_extraction import UAIREFeatureExtractor
from .metadata import dataset_manifest, experiment_manifest, write_json
from .registries import build_backbone, build_dataset
from .schema import STANDARDIZED_COLUMNS
from .validation import validate_dataframe


class DatasetOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config

    def run(self) -> Path:
        self._seed_everything()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_new_run(output_dir)
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dataset, class_count = build_dataset(
            self.config.dataset, self.config.data_dir, self.config.split,
            self.config.download, self.config.normalize, self.config.image_size,
        )
        if self.config.max_samples is not None:
            dataset = Subset(dataset, range(min(self.config.max_samples, len(dataset))))
        model = build_backbone(self.config.backbone, self.config.checkpoint, class_count, device)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        extractor = UAIREFeatureExtractor(model, device)
        rows: list[dict] = []
        try:
            for tensors, labels, source_indices, originals in loader:
                for tensor, label, source_index, original in zip(tensors, labels, source_indices, originals):
                    prediction, features = extractor.extract(tensor, original.cpu().numpy())
                    true_label = int(label.item())
                    rows.append({
                        "image_id": f"{self.config.dataset}:{self.config.split}:{int(source_index.item())}",
                        "dataset": self.config.dataset,
                        "split": self.config.split,
                        "model_name": self.config.model_name or self.config.backbone,
                        "architecture": self.config.backbone,
                        "checkpoint_name": Path(self.config.checkpoint).name,
                        "number_of_classes": class_count,
                        "prediction": prediction,
                        "true_label": true_label,
                        "failure_label": int(prediction != true_label),
                        **features,
                    })
        finally:
            extractor.cleanup()
        frame = pd.DataFrame(rows, columns=STANDARDIZED_COLUMNS)
        report = validate_dataframe(frame, self.config)
        write_json(output_dir / "validation_report.json", report)
        if not report["passed"]:
            raise RuntimeError("Dataset validation failed; see validation_report.json")
        frame.to_csv(output_dir / "reliability_dataset.csv", index=False)
        checkpoint = Path(self.config.checkpoint)
        save_config(self.config, output_dir / "configuration.yaml")
        source = dataset.dataset if isinstance(dataset, Subset) else dataset
        write_json(output_dir / "dataset_manifest.json", dataset_manifest(self.config, source, class_count))
        write_json(output_dir / "experiment_manifest.json", experiment_manifest(self.config, checkpoint, STANDARDIZED_COLUMNS))
        write_json(output_dir / "metadata.json", {
            "dataset_file": "reliability_dataset.csv", "row_count": len(frame),
            "column_count": len(frame.columns), "validation_file": "validation_report.json",
        })
        return output_dir

    def _ensure_new_run(self, output_dir: Path) -> None:
        artifacts = [output_dir / name for name in ("reliability_dataset.csv", "configuration.yaml", "dataset_manifest.json", "experiment_manifest.json", "metadata.json", "validation_report.json")]
        present = [str(path) for path in artifacts if path.exists()]
        if present:
            raise FileExistsError(f"Output directory already contains orchestrator artifacts: {present}. Choose a new output_dir.")

    def _seed_everything(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
