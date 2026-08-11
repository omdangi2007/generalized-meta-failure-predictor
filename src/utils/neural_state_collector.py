"""
=========================================================
UAIRE - Universal AI Reliability Engine

Neural State Collector (NSC)

Collects:
1. Logits
2. Activations
3. Gradients (optional)

Works with any PyTorch model.
=========================================================
"""

import torch


class NeuralStateCollector:

    def __init__(self, target_layer):

        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = None
        self.backward_handle = None

    # ----------------------------------------------------
    # Forward Hook
    # ----------------------------------------------------

    def _forward_hook(self, module, input, output):

        self.activations = output.detach()

    # ----------------------------------------------------
    # Backward Hook
    # ----------------------------------------------------

    def _backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    # ----------------------------------------------------
    # Register Hooks
    # ----------------------------------------------------

    def register_hooks(self):

        self.forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    # ----------------------------------------------------
    # Remove Hooks
    # ----------------------------------------------------

    def remove_hooks(self):

        if self.forward_handle is not None:
            self.forward_handle.remove()

        if self.backward_handle is not None:
            self.backward_handle.remove()

    # ----------------------------------------------------
    # Collect Neural State
    # ----------------------------------------------------

    def collect(
        self,
        model,
        image,
        compute_gradients=True
    ):

        self.activations = None
        self.gradients = None

        model.zero_grad()

        output = model(image)

        prediction = torch.argmax(output, dim=1)

        if compute_gradients:

            score = output[
                torch.arange(output.size(0)),
                prediction
            ].sum()

            score.backward()

        return {

            "logits": output,

            "prediction": prediction,

            "activations": self.activations,

            "gradients": self.gradients

        }