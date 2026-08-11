"""Build one universal research dataset from orchestrated reliability datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.dataset_orchestration.config import OrchestratorConfig, load_config
from src.dataset_orchestration.execution import DatasetOrchestrator
from src.dataset_orchestration.metadata import sha256_file, write_json
from src.dataset_orchestration.schema import STANDARDIZED_COLUMNS
from src.dataset_orchestration.validation import validate_dataframe

from .config import MetaDatasetConfig
from .reporting import dataset_statistics, quality_report, write_plots


class UniversalMetaDatasetBuilder:
    def __init__(self, config: MetaDatasetConfig):
        self.config = config

    def run(self) -> Path:
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        source_root = output / "source_datasets"
        source_root.mkdir(exist_ok=True)
        reports, frames = [], []
        for combination in self.config.combinations:
            report, frame = self._generate_or_load(combination, source_root)
            reports.append(report)
            if frame is not None:
                frames.append(frame)
        if not frames:
            write_json(output / "merge_manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": reports, "merged": False})
            raise RuntimeError("No supported reliability dataset combinations completed successfully.")
        merged = pd.concat(frames, axis=0, ignore_index=True)
        merged = merged.loc[:, list(STANDARDIZED_COLUMNS)]
        report = quality_report(merged, reports)
        if report["missing_value_count"] or report["duplicate_columns"] or report["metadata_consistency"]["invalid_source_runs"]:
            write_json(output / "quality_report.json", report)
            raise RuntimeError("Merged dataset quality validation failed; see quality_report.json")
        merged.to_csv(output / "universal_meta_dataset.csv", index=False)
        write_json(output / "quality_report.json", report)
        write_json(output / "dataset_statistics.json", dataset_statistics(merged))
        figures = write_plots(merged, output / "plots")
        write_json(output / "merge_manifest.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema": list(STANDARDIZED_COLUMNS), "total_rows": len(merged),
            "source_count": len(frames), "skipped_source_count": len(reports) - len(frames),
            "sources": reports, "figures": [f"plots/{name}" for name in figures],
        })
        with (output / "meta_dataset_configuration.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config.as_dict(), handle, sort_keys=False)
        return output

    def _generate_or_load(self, combination, source_root: Path) -> tuple[dict, pd.DataFrame | None]:
        run_dir = source_root / combination.id
        dataset_file = run_dir / "reliability_dataset.csv"
        try:
            if not dataset_file.is_file():
                source_config = OrchestratorConfig(
                    dataset=combination.dataset, backbone=combination.backbone,
                    checkpoint=combination.checkpoint, output_dir=str(run_dir), data_dir=self.config.data_dir,
                    split=self.config.split, batch_size=self.config.batch_size, model_name=combination.model_name,
                    device=self.config.device, num_workers=self.config.num_workers, seed=self.config.seed,
                    max_samples=self.config.max_samples, download=self.config.download,
                    normalize=self.config.normalize, image_size=combination.image_size,
                )
                DatasetOrchestrator(source_config).run()
            source_config = load_config(run_dir / "configuration.yaml")
            frame = pd.read_csv(dataset_file)
            validation = validate_dataframe(frame, source_config)
            if not validation["passed"]:
                raise ValueError("source CSV fails standardized schema validation")
            return {
                "id": combination.id, "status": "included", "path": str(run_dir),
                "rows": len(frame), "sha256": sha256_file(dataset_file), "validation_passed": True,
            }, frame
        except Exception as error:
            return {
                "id": combination.id, "status": "skipped", "reason": f"{type(error).__name__}: {error}",
                "path": str(run_dir), "rows": 0, "validation_passed": False,
            }, None
