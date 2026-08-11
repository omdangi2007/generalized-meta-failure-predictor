# Universal Reliability Dataset Orchestrator

`python -m src.dataset_orchestration configs/reliability_dataset.example.yaml` produces one self-contained, reproducible run directory. It does not train a model, change the predictor, modify the Streamlit app, or alter UAIRE extractor code.

## Architecture and pipeline

`configuration -> dataset/backbone registries -> UniversalModelAdapter -> UniversalReliabilityPipeline -> standardized rows -> validation -> artifacts`

The configured loader batches file access, but each item is passed separately to the existing adapter and extractors. This is required because current UAIRE extractors expose scalar `.item()` feature values; it preserves feature definitions rather than changing them for batched execution.

The supported registries initially cover `cifar10`, `cifar100`, and `fashionmnist`, plus `resnet18`, `resnet34`, `mobilenetv2`, `efficientnetb0`, and `vit_b_16`. Add future supported sources at the registry boundary only.

## Output directory

```
results/reliability_datasets/<run>/
  reliability_dataset.csv
  configuration.yaml
  dataset_manifest.json
  experiment_manifest.json
  metadata.json
  validation_report.json
```

The pipeline refuses to overwrite an existing orchestrator run. The experiment manifest hashes the exact checkpoint and records runtime versions; the dataset manifest records the source split, size, and transform.

## CSV schema

Every output always has this exact ordered header:

```text
image_id,dataset,split,model_name,architecture,checkpoint_name,number_of_classes,prediction,true_label,failure_label,confidence,entropy,prediction_margin,activation_mean,activation_std,activation_max,activation_min,activation_energy,activation_sparsity,dead_neuron_ratio,positive_activation_ratio,activation_entropy,gradient_mean,gradient_std,gradient_max,gradient_min,gradient_norm,gradient_energy,gradient_sparsity,gradient_entropy,brightness,contrast,image_entropy,blur_score,laplacian_variance,edge_density,msp_probability,msp_ood_score,energy_score,normalized_energy,mahalanobis_distance,mahalanobis_available
```

The feature names and their UAIRE registry order are unchanged. The metadata columns precede them to provide a single universal schema across backbones.

## Validation

Before writing the CSV, the pipeline verifies exact schema/order, missing and duplicate columns, null/NaN values, finite numeric features, fixed run metadata, unique image IDs, binary failure labels, and that each failure label equals `prediction != true_label`. The JSON validation report is written even if validation fails.
