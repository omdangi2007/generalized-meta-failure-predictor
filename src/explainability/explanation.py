"""Lightweight adapters for existing UAIRE GradCAM, SHAP, and LIME implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .contracts import BaseExplainer


def _as_pil_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("Image explanation requires an RGB image or CHW image tensor.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class GradCAMExplainer(BaseExplainer):
    method_name = "gradcam"

    def supports(self, runtime_context, additional_data=None) -> bool:
        return bool(runtime_context.hooks.gradcam_layer and "gradcam" in runtime_context.capabilities.supported_explanations)

    def explain(self, *, model, runtime_context, input_tensor, prediction=None, additional_data=None):
        from src.xai.gradcam_features import GradCAMExplainer as ExistingGradCAM
        layer = dict(model.named_modules()).get(runtime_context.hooks.gradcam_layer)
        if layer is None:
            raise RuntimeError(f"Runtime GradCAM layer '{runtime_context.hooks.gradcam_layer}' is unavailable.")
        image = _as_pil_image((additional_data or {}).get("image", input_tensor))
        device = next(model.parameters()).device
        result = ExistingGradCAM(model, layer, device).explain(image, class_index=prediction)
        return {
            "heatmap": result["heatmap"], "importance_scores": {"target_class": result["class_index"]},
            "text_summary": "GradCAM highlights image regions that most supported the selected class.",
            "confidence": result["confidence"], "metadata": {"overlay": result["overlay"], "colored_heatmap": result["colored_heatmap"]},
        }

    def metadata(self) -> dict[str, Any]:
        return {"backend": "src.xai.gradcam_features.GradCAMExplainer", "input": "RGB image"}


class SHAPExplainer(BaseExplainer):
    method_name = "shap"

    def supports(self, runtime_context, additional_data=None) -> bool:
        data = additional_data or {}
        return bool(data.get("meta_model") or data.get("tabular") or data.get("features") is not None)

    def explain(self, *, model, runtime_context, input_tensor, prediction=None, additional_data=None):
        from src.xai.shap_features import SHAPMetaExplainer
        data = additional_data or {}
        if data.get("features") is None:
            raise ValueError("SHAP explanation requires additional_data['features'].")
        result = SHAPMetaExplainer().explain(data["features"])
        top = result["top_features"]
        scores = {str(row.feature): float(row.shap_value) for row in top.itertuples()}
        return {
            "heatmap": None, "importance_scores": scores, "text_summary": result["summary"],
            "confidence": float(result["failure_probability"]),
            "metadata": {"base_failure_probability": result["base_failure_probability"], "top_features": top.to_dict(orient="records")},
        }

    def metadata(self) -> dict[str, Any]:
        return {"backend": "src.xai.shap_features.SHAPMetaExplainer", "input": "tabular reliability features"}


class LIMEExplainer(BaseExplainer):
    method_name = "lime"

    def supports(self, runtime_context, additional_data=None) -> bool:
        return "lime" in runtime_context.capabilities.supported_explanations

    def explain(self, *, model, runtime_context, input_tensor, prediction=None, additional_data=None):
        from src.xai.lime_features import LIMEImageExplainer
        image = _as_pil_image((additional_data or {}).get("image", input_tensor))
        device = next(model.parameters()).device
        result = LIMEImageExplainer(model, device).explain(image, class_index=prediction)
        return {
            "heatmap": result["positive_mask"],
            "importance_scores": {f"superpixel_{item['superpixel']}": item["weight"] for item in result["top_superpixels"]},
            "text_summary": "LIME identifies superpixels that locally support or contradict the prediction.",
            "confidence": result["confidence"],
            "metadata": {"top_superpixels": result["top_superpixels"], "positive_overlay": result["positive_overlay"], "signed_overlay": result["signed_overlay"]},
        }

    def metadata(self) -> dict[str, Any]:
        return {"backend": "src.xai.lime_features.LIMEImageExplainer", "input": "RGB image"}
