from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from semantic_kv.collectors.jsonl_collector import JsonlTraceCollector
from semantic_kv.collectors.mock_backend import MockLLMBackend
from semantic_kv.traces.io import load_trace


def test_jsonl_collector_writes_loadable_trace(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    backend = MockLLMBackend()
    collector = JsonlTraceCollector(path, backend_name="test_backend", model_name=backend.model_name, drop_raw_text=False)
    try:
        with collector.trace_request(
            prompt="SYSTEM: hello\n\nUSER: summarize the shared policy",
            model_name=backend.model_name,
            session_id="s1",
            tenant_id="t1",
        ) as span:
            response = backend.generate("SYSTEM: hello\n\nUSER: summarize the shared policy")
            span.set_observed_output_tokens(response["output_tokens"])
            span.set_observed_prefix_hit_tokens(response["observed_prefix_hit_tokens"])
            span.set_observed_prefill_tokens(response["observed_prefill_tokens"])
            span.set_backend_metadata(response["metadata"])
    finally:
        collector.close()
    trace = load_trace(path)
    assert len(trace.requests) == 1
    assert trace.requests[0].blocks
    assert trace.requests[0].observed_ttft_ms is not None


def test_jsonl_collector_can_drop_raw_text(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    collector = JsonlTraceCollector(path, backend_name="test_backend", drop_raw_text=True)
    try:
        with collector.trace_request(prompt="USER: private text", model_name="mock") as span:
            span.set_backend_metadata({"observed_ttft_ms": 1.0})
    finally:
        collector.close()
    trace = load_trace(path)
    block = trace.requests[0].blocks[0]
    assert block.text is None
    assert block.content_hash
    assert block.token_count > 0


def test_mock_backend_is_deterministic_for_fresh_instances():
    prompt = "SYSTEM: stable\n\nUSER: stable request"
    first = MockLLMBackend().generate(prompt)
    second = MockLLMBackend().generate(prompt)
    assert first == second


def test_collect_mock_trace_cli_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_mock_trace.py",
            "--requests",
            "5",
            "--output-name",
            "collector_smoke",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "SemanticKV mock trace collection complete" in result.stdout
    traces = list(tmp_path.glob("*_collector_smoke.jsonl"))
    assert traces
    assert len(load_trace(traces[0]).requests) == 5


def test_fastapi_mock_generate_endpoint():
    client = TestClient(app)
    response = client.post(
        "/v1/mock/generate",
        json={"prompt": "SYSTEM: demo\n\nUSER: hello", "session_id": "api-session"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "response" in payload
    assert payload["trace_path"].endswith("_mock_backend_trace.jsonl")

