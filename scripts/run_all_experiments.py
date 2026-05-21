#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_kv.cache.models import ExperimentConfig, ExperimentResult
from semantic_kv.experiments.runner import ExperimentRunner, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all SemanticKV evaluation configs.")
    parser.add_argument("--config-dir", default="configs/experiments")
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    config_paths = sorted(Path(args.config_dir).glob("*.yaml"))
    if not config_paths:
        raise SystemExit(f"No experiment configs found in {args.config_dir}")

    print(f"SemanticKV aggregate evaluation")
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Experiments: {len(config_paths)}")

    results: list[ExperimentResult] = []
    for index, config_path in enumerate(config_paths, start=1):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["output_dir"] = args.output_dir
        config = ExperimentConfig(**raw)
        print(f"\n[{index}/{len(config_paths)}] Running {config.experiment_id} from {config_path}")
        result = ExperimentRunner(config).run(write_outputs=True)
        results.append(result)
        print(summary_table(result))
        print(f"Outputs: {result.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_aggregate_csv(output_dir / "aggregate_summary.csv", results)
    write_aggregate_markdown(output_dir / "aggregate_summary.md", results)
    print("\nAggregate adaptive vs LRU summary:")
    print(_short_adaptive_summary(results))
    print(f"\nAggregate CSV: {output_dir / 'aggregate_summary.csv'}")
    print(f"Aggregate Markdown: {output_dir / 'aggregate_summary.md'}")


def adaptive_vs_lru_row(result: ExperimentResult) -> dict[str, object]:
    adaptive = result.metrics_by_policy.get("adaptive_semantic")
    lru = result.metrics_by_policy.get("lru")
    if adaptive is None or lru is None:
        return {
            "experiment": result.experiment_id,
            "best_policy_by_avg_ttft": result.best_policy_by_ttft,
            "adaptive_avg_ttft": "",
            "lru_avg_ttft": "",
            "adaptive_relative_ttft_improvement_vs_lru": "",
            "adaptive_hit_rate": "",
            "lru_hit_rate": "",
            "adaptive_relative_hit_rate_improvement_vs_lru": "",
        }
    return {
        "experiment": result.experiment_id,
        "best_policy_by_avg_ttft": result.best_policy_by_ttft,
        "adaptive_avg_ttft": round(adaptive.estimated_ttft_ms_avg, 4),
        "lru_avg_ttft": round(lru.estimated_ttft_ms_avg, 4),
        "adaptive_relative_ttft_improvement_vs_lru": round(adaptive.relative_ttft_improvement_vs_lru or 0.0, 6),
        "adaptive_hit_rate": round(adaptive.hit_rate, 6),
        "lru_hit_rate": round(lru.hit_rate, 6),
        "adaptive_relative_hit_rate_improvement_vs_lru": round(adaptive.relative_hit_rate_improvement_vs_lru or 0.0, 6),
    }


def write_aggregate_csv(path: Path, results: list[ExperimentResult]) -> None:
    rows = [adaptive_vs_lru_row(result) for result in results]
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_markdown(path: Path, results: list[ExperimentResult]) -> None:
    lines = [
        "# SemanticKV Aggregate Evaluation Summary",
        "",
        "All TTFT values are simulated estimates, not measured GPU latency.",
        "",
        "| Experiment | Best Avg TTFT Policy | Adaptive Avg TTFT | LRU Avg TTFT | Adaptive TTFT Improvement vs LRU | Adaptive Hit Rate | LRU Hit Rate | Adaptive Hit-Rate Improvement vs LRU |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        row = adaptive_vs_lru_row(result)
        lines.append(
            f"| {row['experiment']} | {row['best_policy_by_avg_ttft']} | {row['adaptive_avg_ttft']} | "
            f"{row['lru_avg_ttft']} | {float(row['adaptive_relative_ttft_improvement_vs_lru'] or 0) * 100:.2f}% | "
            f"{row['adaptive_hit_rate']} | {row['lru_hit_rate']} | "
            f"{float(row['adaptive_relative_hit_rate_improvement_vs_lru'] or 0) * 100:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _short_adaptive_summary(results: list[ExperimentResult]) -> str:
    rows = ["experiment                         adaptive_vs_lru_ttft  best_policy"]
    for result in results:
        row = adaptive_vs_lru_row(result)
        improvement = float(row["adaptive_relative_ttft_improvement_vs_lru"] or 0.0) * 100
        rows.append(f"{str(row['experiment']):<34} {improvement:>8.2f}%          {row['best_policy_by_avg_ttft']}")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
