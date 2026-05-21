#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_kv.traces.io import load_trace
from semantic_kv.traces.models import LatencyModelConfig, TraceReplayConfig
from semantic_kv.traces.replay import TraceReplayRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a serving-style trace through SemanticKV policies.")
    parser.add_argument("--trace", required=True, help="Path to .jsonl, .json, or .csv trace")
    parser.add_argument("--cache-token-budget", type=int, required=True)
    parser.add_argument("--policies", nargs="+", default=["lru", "adaptive_semantic", "static_semantic"])
    parser.add_argument("--output-name", default="trace_replay")
    parser.add_argument("--output-dir", default="outputs/traces")
    parser.add_argument("--base-ms", type=float, default=80.0)
    parser.add_argument("--ms-per-uncached-token", type=float, default=0.08)
    parser.add_argument("--calibrate-from-trace", action="store_true")
    args = parser.parse_args()

    trace = load_trace(args.trace)
    print("SemanticKV trace replay")
    print(f"Trace: {args.trace}")
    print(f"Trace ID: {trace.trace_id}")
    print(f"Backend label: {trace.backend_name}")
    print(f"Requests: {len(trace.requests)}")
    print(f"Policies: {', '.join(args.policies)}")
    print(f"Cache token budget: {args.cache_token_budget}")
    print("Latency model: calibrated from observed TTFT" if args.calibrate_from_trace else "Latency model: configured linear estimate")

    config = TraceReplayConfig(
        trace_path=args.trace,
        cache_token_budget=args.cache_token_budget,
        policies=args.policies,
        latency_model=LatencyModelConfig(
            base_ms=args.base_ms,
            ms_per_uncached_token=args.ms_per_uncached_token,
        ),
        output_name=args.output_name,
        calibrate_from_trace=args.calibrate_from_trace,
        output_dir=args.output_dir,
    )
    result = TraceReplayRunner(config).run(write_outputs=True)
    print("\nPolicy replay summary:")
    print(_table(result))
    print(f"\nOutputs: {result['output_dir']}")
    print("Note: results are offline simulated/calibrated trace replay, not deployed backend measurements.")


def _table(result: dict) -> str:
    rows = ["policy                hit_rate  tokens_saved  avg_ttft  p95_ttft  evictions"]
    for policy, metric in sorted(result["metrics_by_policy"].items(), key=lambda item: item[1]["estimated_ttft_ms_avg"]):
        rows.append(
            f"{policy:<21} {metric['hit_rate']:>7.3f}  {metric['tokens_saved_total']:>12}  "
            f"{metric['estimated_ttft_ms_avg']:>8.2f}  {metric['estimated_ttft_ms_p95']:>8.2f}  {metric['evictions_total']:>9}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()

