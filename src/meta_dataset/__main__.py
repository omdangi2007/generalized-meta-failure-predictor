"""CLI: python -m src.meta_dataset configs/universal_meta_dataset.yaml"""

from __future__ import annotations

import argparse

from .config import load_meta_config
from .construction import UniversalMetaDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the UAIRE Universal Meta Dataset.")
    parser.add_argument("config", help="Path to the YAML meta-dataset configuration.")
    arguments = parser.parse_args()
    print(UniversalMetaDatasetBuilder(load_meta_config(arguments.config)).run())


if __name__ == "__main__":
    main()
