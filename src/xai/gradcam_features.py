"""
=========================================================
UAIRE

GradCAM Visual Explainer

Generates visual attention maps for the backbone model
without modifying the backbone architecture.
=========================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.image_preprocessor import ImagePreprocessor


class GradCAMExplainer:

    """
    Computes GradCAM for a PyTorch classification backbone.

    Hooks are registered only during explanation and removed
    immediately afterward, so the model architecture and weights
    remain unchanged.
    """

    def __init__(self, model, target_layer, device):

        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.preprocessor = ImagePreprocessor()

    def explain(self, image, class_index=None, alpha=0.42):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        original = np.asarray(image).astype(np.float32) / 255.0
        input_tensor = self.preprocessor.process(image).to(self.device)

        activations = []
        gradients = []

        def forward_hook(module, module_input, module_output):
            activations.append(module_output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        forward_handle = self.target_layer.register_forward_hook(
            forward_hook
        )
        backward_handle = self.target_layer.register_full_backward_hook(
            backward_hook
        )

        try:
            self.model.zero_grad(set_to_none=True)
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            if class_index is None:
                class_index = int(torch.argmax(probabilities, dim=1).item())

            confidence = float(probabilities[0, class_index].item())
            score = output[0, class_index]
            score.backward()

            if not activations or not gradients:
                raise RuntimeError("GradCAM hooks did not capture activations or gradients.")

            activation = activations[-1]
            gradient = gradients[-1]

            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(original.shape[0], original.shape[1]),
                mode="bilinear",
                align_corners=False,
            )

            heatmap = cam.squeeze().detach().cpu().numpy()
            heatmap = heatmap - heatmap.min()
            max_value = heatmap.max()
            if max_value > 0:
                heatmap = heatmap / max_value

            colored_heatmap = self._colorize_heatmap(heatmap)
            overlay = np.clip(
                (1.0 - alpha) * original + alpha * colored_heatmap,
                0,
                1,
            )

            return {
                "class_index": class_index,
                "confidence": confidence,
                "heatmap": (heatmap * 255).astype(np.uint8),
                "colored_heatmap": (colored_heatmap * 255).astype(np.uint8),
                "overlay": (overlay * 255).astype(np.uint8),
            }

        finally:
            forward_handle.remove()
            backward_handle.remove()
            self.model.zero_grad(set_to_none=True)

    def _colorize_heatmap(self, heatmap):

        heatmap = np.clip(heatmap, 0, 1)

        red = np.clip(1.8 * heatmap, 0, 1)
        green = np.clip(1.8 * (1 - np.abs(heatmap - 0.55) * 2), 0, 1)
        blue = np.clip(1.4 * (1 - heatmap), 0, 1) * 0.35

        return np.stack([red, green, blue], axis=-1)
