"""
=========================================================
UAIRE

Model Loader

Loads trained backbone models.
=========================================================
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.config.paths import RESNET18_CIFAR10


def load_resnet18_cifar10(device="cpu"):

    # -------------------------------------------------
    # Locate project root
    # -------------------------------------------------

    from src.config.paths import RESNET18_CIFAR10

    checkpoint = RESNET18_CIFAR10

    # -------------------------------------------------
    # Build model
    # -------------------------------------------------

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