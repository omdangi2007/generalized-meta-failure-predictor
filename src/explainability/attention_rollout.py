"""Runtime-aware attention-rollout interface with a replaceable placeholder backend."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .contracts import BaseExplainer


class AttentionRolloutExplainer(BaseExplainer):
    """Captures final transformer activations; no research-grade rollout is claimed."""

    method_name = "attention_rollout"

    def supports(self, runtime_context, additional_data=None) -> bool:
        return bool(runtime_context.metadata.transformer_blocks and runtime_context.hooks.feature_layer)

    def explain(self, *, model, runtime_context, input_tensor, prediction=None, additional_data=None):
        layer = dict(model.named_modules()).get(runtime_context.hooks.feature_layer)
        if layer is None:
            raise RuntimeError("Transformer feature layer selected by runtime is unavailable.")
        captured = []

        def hook(_, __, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        handle = layer.register_forward_hook(hook)
        try:
            tensor = input_tensor if isinstance(input_tensor, torch.Tensor) else (additional_data or {}).get("model_input")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Attention rollout requires an input tensor or additional_data['model_input'].")
            device = next(model.parameters()).device
            with torch.no_grad():
                logits = model(tensor.to(device))
                probabilities = torch.softmax(logits, dim=1)
            if not captured:
                raise RuntimeError("Transformer hook did not capture activations.")
            heatmap = self._placeholder_heatmap(captured[-1])
            chosen = int(torch.argmax(probabilities, dim=1).item()) if prediction is None else int(prediction)
            return {
                "heatmap": heatmap,
                "importance_scores": {"token_activation_mean": float(heatmap.mean())},
                "text_summary": "Placeholder attention visualization from the final transformer feature block; it is not research-grade attention rollout.",
                "confidence": float(probabilities[0, chosen].item()),
                "metadata": {"hook_layer": runtime_context.hooks.feature_layer, "placeholder": True, "target_class": chosen},
                "warnings": ("Attention rollout backend is a replaceable placeholder based on final-block activations.",),
            }
        finally:
            handle.remove()

    @staticmethod
    def _placeholder_heatmap(activation) -> np.ndarray:
        values = activation.detach().float().abs()
        if values.ndim == 3:
            values = values[0].mean(dim=-1)
        else:
            values = values.flatten()
        values = values.detach().cpu().numpy().reshape(-1)
        if len(values) > 1 and int(math.sqrt(len(values) - 1)) ** 2 == len(values) - 1:
            values = values[1:]
        side = max(1, int(math.sqrt(len(values))))
        values = np.pad(values[: side * side], (0, max(0, side * side - len(values))))
        heatmap = values.reshape(side, side)
        return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

    def metadata(self) -> dict[str, Any]:
        return {"backend": "runtime activation hook placeholder", "research_grade": False}
