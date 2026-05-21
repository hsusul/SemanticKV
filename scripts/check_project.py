#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import semantic_kv
    from apps.api.main import app
    from semantic_kv.policies.adaptive_semantic import AdaptiveSemanticPolicy
    from semantic_kv.policies.fifo import FIFOPolicy
    from semantic_kv.policies.lfu import LFUPolicy
    from semantic_kv.policies.lru import LRUPolicy
    from semantic_kv.policies.size_aware_lru import SizeAwareLRUPolicy
    from semantic_kv.policies.static_semantic import StaticSemanticPolicy
    from semantic_kv.latency.models import CalibratedLatencyModel, LinearLatencyModel
    from semantic_kv.traces.replay import TraceReplayRunner

    config_dir = Path("configs/experiments")
    if not config_dir.exists():
        raise SystemExit("configs/experiments does not exist")
    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit("configs/experiments contains no YAML configs")

    policies = [
        FIFOPolicy(),
        LRUPolicy(),
        LFUPolicy(),
        SizeAwareLRUPolicy(),
        StaticSemanticPolicy(),
        AdaptiveSemanticPolicy(),
    ]
    policy_names = ", ".join(policy.name() for policy in policies)
    routes = {route.path for route in app.routes}
    if "/health" not in routes:
        raise SystemExit("FastAPI app missing /health route")
    trace_dir = Path("examples/traces")
    required_traces = [
        trace_dir / "small_serving_trace.jsonl",
        trace_dir / "rag_serving_trace.jsonl",
        trace_dir / "workload_shift_trace.jsonl",
    ]
    missing = [str(path) for path in required_traces if not path.exists()]
    if missing:
        raise SystemExit(f"Missing example traces: {missing}")
    _ = LinearLatencyModel()
    _ = CalibratedLatencyModel()
    _ = TraceReplayRunner

    print("SemanticKV project check passed")
    print(f"Package version: {semantic_kv.__version__}")
    print(f"Experiment configs: {len(configs)}")
    print(f"Policies: {policy_names}")
    print(f"Example traces: {len(required_traces)}")
    print("Trace replay imports: ok")
    print("FastAPI app import: ok")


if __name__ == "__main__":
    main()
