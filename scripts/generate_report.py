#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_kv.cache.models import ExperimentResult
from semantic_kv.experiments.reports import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate summary.md from a SemanticKV results.json file.")
    parser.add_argument("results_json")
    args = parser.parse_args()

    path = Path(args.results_json)
    result = ExperimentResult.model_validate_json(path.read_text(encoding="utf-8"))
    write_report(path.parent / "summary.md", result)
    print(path.parent / "summary.md")


if __name__ == "__main__":
    main()
