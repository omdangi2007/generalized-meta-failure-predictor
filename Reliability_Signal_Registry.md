# Universal AI Reliability Engine (UAIRE)

## Reliability Signal Registry

This document maintains the complete list of reliability signals extracted by the UAIRE framework.

---

# Category 1 — Prediction Confidence

| Signal | Status | Description |
|---------|--------|-------------|
| Confidence Score | ✅ Implemented | Maximum predicted probability. |
| Entropy | ✅ Implemented | Prediction uncertainty. |
| Prediction Margin | ✅ Implemented| Difference between top-1 and top-2 probabilities. |

---

# Category 2 — Internal Network Health

| Signal | Status | Description |
|---------|--------|-------------|
| Activation Mean | ✅ Implemented | Mean activation of final feature map. |
| Activation Std | ✅ Implemented | Activation variability. |
| Activation Max | ✅ Implemented | Maximum activation. |
| Activation Min | ✅ Implemented | Minimum activation. |
| Gradient Norm | ⏳ Planned | Magnitude of gradients. |
| Gradient Variance | ⏳ Planned | Gradient dispersion. |
| Gradient Sparsity | ⏳ Planned | Percentage of near-zero gradients. |

---

# Category 3 — Input Quality

| Signal | Status | Description |
|---------|--------|-------------|
| Image Entropy | ⏳ Planned | Complexity of image information. |
| Blur Score | ⏳ Planned | Sharpness estimation. |
| Brightness | ⏳ Planned | Average image intensity. |
| Contrast | ⏳ Planned | Intensity variation. |
| Noise Level | ⏳ Planned | Estimated image noise. |

---

# Category 4 — Distribution Awareness

| Signal | Status | Description |
|---------|--------|-------------|
| Mahalanobis Distance | ⏳ Planned | Distance from training feature distribution. |
| Energy Score | ⏳ Planned | Confidence under distribution shift. |
| Embedding Distance | ⏳ Planned | Distance in latent feature space. |

---

# Category 5 — Explainability

| Signal | Status | Description |
|---------|--------|-------------|
| Grad-CAM Entropy | ⏳ Planned | Spread of model attention. |
| Grad-CAM Concentration | ⏳ Planned | Focus of important regions. |
| SHAP Variance | ⏳ Planned | Stability of feature importance. |
| SHAP Magnitude | ⏳ Planned | Overall explanation strength. |
| LIME Stability | ⏳ Planned | Consistency across perturbations. |

---

# Category 6 — Meta Intelligence

| Signal | Status | Description |
|---------|--------|-------------|
| Failure Probability | ⏳ Planned | Output of meta-model. |
| Reliability Score | ⏳ Planned | Overall trustworthiness score. |
| Failure Reason | ⏳ Planned | Human-readable explanation. |
| Recommendation | ⏳ Planned | Suggested action. |