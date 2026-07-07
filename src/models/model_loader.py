"""
=========================================================
UAIRE

Model Loader

Loads trained backbone models.
=========================================================
"""

import torch
import torch.nn as nn
from torchvision import models


def load_resnet18_cifar10(
    checkpoint="../models/resnet18_cifar10.pth",
    device="cpu"
):
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    state_dict = torch.load(
        checkpoint,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model