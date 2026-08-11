# Universal Meta Dataset

Build the merged research dataset with:

```bash
python -m src.meta_dataset configs/universal_meta_dataset.example.yaml
```

The builder invokes the Universal Reliability Dataset Orchestrator for each available, registered checkpoint/dataset pairing. It validates each source against the immutable UAIRE schema before concatenating rows. Failed or unsupported combinations are not fatal: they are recorded with their failure reason in `merge_manifest.json` and excluded without changing successful datasets.

## Registered combinations

- ResNet-18: CIFAR-10, CIFAR-100, Fashion-MNIST
- ResNet-34: CIFAR-10
- MobileNetV2: CIFAR-10
- EfficientNet-B0: CIFAR-10
- ViT-B/16: CIFAR-10 at 224×224, only when its supplied checkpoint runs successfully

## Outputs

```
results/meta_dataset/
  universal_meta_dataset.csv
  dataset_statistics.json
  merge_manifest.json
  quality_report.json
  meta_dataset_configuration.yaml
  plots/
  source_datasets/
```

The master CSV preserves the exact 42-column orchestrator schema: metadata (`image_id`, dataset, split, model and architecture identity, `checkpoint_name`, class count, prediction, true label, and failure label) followed by the existing UAIRE features in their unchanged order. `checkpoint_name` is the standardized checkpoint metadata field; the full checkpoint path and SHA-256 are retained in each source experiment manifest.

Quality reporting covers row totals, architecture/dataset counts, feature count, missing values, duplicate rows and columns, metadata validity, failure-label distribution, and true-class distribution. Statistics include architecture/dataset counts, failure rates, and basic statistics for every UAIRE feature. The plot set contains architecture, dataset, and failure distributions, feature correlation, and missingness.

## Known limitations

- Source extraction remains single-sample internally to preserve existing scalar extractor behavior.
- A combination is included only if its checkpoint, architecture, dataset files, and standardized validation all succeed. Skipped combinations are recorded, not repaired or retrained.
- Mahalanobis availability remains determined by the existing UAIRE detector/statistics compatibility.
- No model training, predictor/UI change, adapter change, or feature-extractor change is performed.
