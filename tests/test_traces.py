from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from semantic_kv.latency.models import CalibratedLatencyModel
from semantic_kv.traces.anonymize import anonymize_trace
from semantic_kv.traces.io import load_trace
from semantic_kv.traces.models import TraceReplayConfig
from semantic_kv.traces.replay import TraceReplayRunner


def test_jsonl_trace_loading():
    trace = load_trace("examples/traces/rag_serving_trace.jsonl")
    assert trace.trace_id == "rag_serving_trace"
    assert len(trace.requests) == 30
    assert trace.requests[0].blocks
    assert trace.requests[0].observed_ttft_ms is not None


def test_trace_anonymization_drops_text_and_preserves_counts():
    trace = load_trace("examples/traces/small_serving_trace.jsonl")
    anonymized = anonymize_trace(trace, drop_text=True)
    original_block = trace.requests[0].blocks[0]
    anonymized_block = anonymized.requests[0].blocks[0]
    assert anonymized_block.text is None
    assert anonymized_block.token_count == original_block.token_count
    assert anonymized_block.semantic_type == original_block.semantic_type
    assert anonymized_block.content_hash


def test_calibrated_latency_model_fit_is_deterministic():
    trace = load_trace("examples/traces/rag_serving_trace.jsonl")
    model = CalibratedLatencyModel().fit(trace.requests)
    assert model.samples_used == len(trace.requests)
    assert 60 <= model.base_ms <= 70
    assert 0.06 <= model.ms_per_uncached_token <= 0.09
    assert model.predict(100) == model.predict(100)


def test_trace_replay_writes_outputs(tmp_path: Path):
    config = TraceReplayConfig(
        trace_path="examples/traces/rag_serving_trace.jsonl",
        cache_token_budget=50000,
        policies=["lru", "adaptive_semantic"],
        output_name="test_replay",
        output_dir=str(tmp_path),
        calibrate_from_trace=True,
    )
    result = TraceReplayRunner(config).run(write_outputs=True)
    out = Path(result["output_dir"])
    assert (out / "replay_results.json").exists()
    assert (out / "replay_metrics.csv").exists()
    assert (out / "replay_summary.md").exists()
    assert (out / "replay_policy_avg_ttft.png").exists()
    assert result["latency_model"]["kind"] == "calibrated_linear"


def test_trace_replay_is_deterministic_without_writing():
    config = TraceReplayConfig(
        trace_path="examples/traces/workload_shift_trace.jsonl",
        cache_token_budget=50000,
        policies=["lru", "adaptive_semantic", "static_semantic"],
        calibrate_from_trace=True,
    )
    first = TraceReplayRunner(config).run(write_outputs=False)
    second = TraceReplayRunner(config).run(write_outputs=False)
    assert first["metrics_by_policy"] == second["metrics_by_policy"]


def test_replay_trace_cli_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/replay_trace.py",
            "--trace",
            "examples/traces/rag_serving_trace.jsonl",
            "--cache-token-budget",
            "50000",
            "--policies",
            "lru",
            "adaptive_semantic",
            "--output-name",
            "cli_smoke",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "SemanticKV trace replay" in result.stdout
    assert "Policy replay summary" in result.stdout
    assert list(tmp_path.glob("*_cli_smoke/replay_results.json"))

