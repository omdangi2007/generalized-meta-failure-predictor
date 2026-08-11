# Universal Meta-Model Training

Train one generalized UAIRE failure-prediction model from the immutable Phase 3 dataset:

```bash
python -m src.meta_model_training configs/universal_meta_model.example.yaml
```

The trainer accepts only `results/meta_dataset/universal_meta_dataset.csv` (or an explicitly configured Phase 3 output). Before training, it requires the exact standardized schema and feature order, complete required metadata, finite feature values, a binary `failure_label`, and agreement between failure labels and `prediction != true_label`.

Metadata is preserved in the input and split metadata but excluded from model inputs. The unchanged 32 UAIRE features are used in their standardized order. The target is `failure_label`.

The training procedure uses reproducible, stratified train/validation/test splits and one `RandomForestClassifier` with reasonable recorded hyperparameters. Random Forest does not require feature scaling, so no scaler artifact is generated; `model_metadata.json` records this explicitly.

## Outputs

```
results/meta_model/
  model.pkl
  feature_order.pkl
  training_configuration.yaml
  model_metadata.json
  training_statistics.json
  split_metadata.json
  classification_report.json
  feature_importance.csv
  explainability_summary.md
  PHASE_4_REVIEW_REPORT.md
  plots/
```

Test evaluation includes accuracy, precision, recall, F1, ROC-AUC, classification report, confusion matrix, ROC curve, precision-recall curve, and calibration curve. Explainability outputs include Random Forest importances, a top-20 plot, SHAP summary/bar plots when the runtime permits them, and a natural-language interpretation.

## Limitations

- Training requires an existing Phase 3 master CSV; this phase never replaces or edits it.
- Both target classes need adequate support for reproducible stratified splits.
- Feature importance and SHAP describe predictive association, not causal effect.
- Cross-architecture held-out evaluation is intentionally outside this phase.
