"""CLI: python -m src.dataset_orchestration path/to/config.yaml"""

from __future__ import annotations

import argparse

from .config import load_config
from .execution import DatasetOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standardized UAIRE reliability dataset.")
    parser.add_argument("config", help="Path to a YAML or JSON orchestrator configuration.")
    args = parser.parse_args()
    print(DatasetOrchestrator(load_config(args.config)).run())


if __name__ == "__main__":
    main()
