"""
=========================================================
UAIRE

Universal Model Adapter

Architecture boundary between UAIRE and any PyTorch
classification backbone.
=========================================================
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class UniversalModelAdapter:

    """
    Adapter for PyTorch classification backbones.

    Responsibilities:
    - Own the model/device boundary.
    - Register forward and backward hooks.
    - Capture logits, activations and gradients.
    - Expose a stable interface to UAIRE.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        target_layer: Optional[nn.Module] = None,
    ):

        if model is None:
            raise ValueError(
                "UniversalModelAdapter requires a PyTorch nn.Module."
            )

        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be an instance of torch.nn.Module."
            )

        self.model = model
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

        self.target_layer = target_layer or self._infer_target_layer()

        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None

        self.register_hooks()

    def _resolve_device(self, device: Optional[str]) -> str:

        if device is not None:
            return device

        try:
            parameter = next(self.model.parameters())
            return str(parameter.device)
        except StopIteration:
            pass

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _infer_target_layer(self) -> nn.Module:

        for attribute in ("layer4", "features", "blocks", "stages"):
            if hasattr(self.model, attribute):
                candidate = getattr(self.model, attribute)
                target = self._last_module(candidate)
                if target is not None:
                    return target

        leaf_modules = [
            module
            for module in self.model.modules()
            if module is not self.model
            and len(list(module.children())) == 0
        ]

        for module in reversed(leaf_modules):
            if isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.Linear,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.LayerNorm,
                ),
            ):
                return module

        if leaf_modules:
            return leaf_modules[-1]

        raise ValueError(
            "Could not infer a hook target layer. Pass target_layer explicitly."
        )

    def _last_module(self, module: nn.Module) -> Optional[nn.Module]:

        children = list(module.children())

        if not children:
            return module

        return children[-1]

    def _forward_hook(self, module, inputs, output):

        if isinstance(output, (tuple, list)):
            output = output[0]

        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):

        gradient = grad_output[0]

        if isinstance(gradient, (tuple, list)):
            gradient = gradient[0]

        self.gradients = gradient.detach()

    def register_hooks(self) -> None:

        self.remove_hooks()

        self.forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def remove_hooks(self) -> None:

        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def collect(
        self,
        input_tensor: torch.Tensor,
        compute_gradients: bool = True,
    ) -> dict:

        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        prediction = torch.argmax(logits, dim=1)

        if compute_gradients:
            score = logits[
                torch.arange(logits.size(0), device=logits.device),
                prediction,
            ].sum()
            score.backward()

        return {
            "logits": logits,
            "prediction": prediction,
            "activations": self.activations,
            "gradients": self.gradients,
        }

    def cleanup(self) -> None:

        self.remove_hooks()
