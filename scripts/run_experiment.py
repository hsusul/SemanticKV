#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_kv.cache.models import ExperimentConfig
from semantic_kv.experiments.runner import ExperimentRunner, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SemanticKV policy comparison experiment.")
    parser.add_argument("--config", default="configs/default_experiment.yaml")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = ExperimentConfig(**raw)
    result = ExperimentRunner(config).run(write_outputs=True)
    print(summary_table(result))
    print(f"\nOutputs: {result.output_dir}")


if __name__ == "__main__":
    main()
