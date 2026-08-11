"""
=========================================================
UAIRE

Project Paths

Centralized project directories used across the framework.
=========================================================
"""

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Common Directories
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "results"
FEATURE_DIR = PROJECT_ROOT / "features"

# Common Files
UAIRE_META_MODEL = MODEL_DIR / "uaire_meta_model.pkl"
UAIRE_FEATURE_ORDER = MODEL_DIR / "uaire_feature_order.pkl"

RESNET18_CIFAR10 = MODEL_DIR / "resnet18_cifar10.pth"
RESNET18_CIFAR100 = MODEL_DIR / "resnet18_cifar100.pth"
RESNET34_CIFAR10 = MODEL_DIR / "resnet34_cifar10.pth"
MOBILENETV2_CIFAR10 = MODEL_DIR / "mobilenetv2_cifar10.pth"
EFFICIENTNETB0_CIFAR10 = MODEL_DIR / "efficientnetb0_cifar10.pth"
VIT_CIFAR10 = MODEL_DIR / "vit_cifar10.pth"

MAHALANOBIS_STATS = MODEL_DIR / "mahalanobis_stats.npz"