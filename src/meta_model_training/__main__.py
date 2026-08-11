"""CLI: python -m src.meta_model_training configs/universal_meta_model.yaml"""

from __future__ import annotations

import argparse

from .config import load_training_config
from .training import UniversalMetaModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one generalized UAIRE meta-model.")
    parser.add_argument("config", help="Path to YAML training configuration.")
    args = parser.parse_args()
    print(UniversalMetaModelTrainer(load_training_config(args.config)).run())


if __name__ == "__main__":
    main()
