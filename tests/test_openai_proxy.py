from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from apps.proxy.main import app
import apps.proxy.main as proxy_main
from semantic_kv.adapters.openai_proxy.client import OpenAICompatibleClient
from semantic_kv.adapters.openai_proxy.models import OpenAIProxySettings
from semantic_kv.adapters.openai_proxy.proxy import OpenAICompatibleTelemetryProxy
from semantic_kv.traces.io import load_trace
from semantic_kv.traces.models import TraceReplayConfig
from semantic_kv.traces.replay import TraceReplayRunner


def _install_proxy(tmp_path: Path, redact_text: bool = True) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        if payload.get("stream"):
            content = (
                'data: {"id":"chunk-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"hello"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=content.encode("utf-8"), headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            },
        )

    transport = httpx.MockTransport(handler)
    settings = OpenAIProxySettings(
        upstream_base_url="http://mock-upstream",
        trace_output_dir=str(tmp_path),
        redact_text=redact_text,
        timeout_seconds=5,
    )
    proxy_main.proxy = OpenAICompatibleTelemetryProxy(
        settings,
        client=OpenAICompatibleClient("http://mock-upstream", timeout_seconds=5, transport=transport),
    )


def test_proxy_health_endpoint(tmp_path: Path):
    _install_proxy(tmp_path)
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "semantic-kv-openai-proxy"


def test_non_streaming_proxy_forwards_and_records_trace(tmp_path: Path):
    _install_proxy(tmp_path, redact_text=True)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": "sess-1", "X-Tenant-ID": "tenant-a"},
        json={
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "You are a proxy test."},
                {"role": "user", "content": "CONTEXT: shared document\n\nUSER: answer"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-test"
    trace_path = next(tmp_path.glob("*_openai_proxy_trace.jsonl"))
    trace = load_trace(trace_path)
    request = trace.requests[0]
    assert request.session_id == "sess-1"
    assert request.tenant_id == "tenant-a"
    assert request.observed_ttft_ms is not None
    assert request.metadata["adapter"] == "openai_proxy"
    assert all(block.text is None for block in request.blocks)


def test_streaming_proxy_records_ttft_and_trace_is_replayable(tmp_path: Path):
    _install_proxy(tmp_path, redact_text=False)
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "stream": True,
            "messages": [{"role": "user", "content": "USER: stream please"}],
        },
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    assert "data:" in body
    trace_path = next(tmp_path.glob("*_openai_proxy_trace.jsonl"))
    trace = load_trace(trace_path)
    assert trace.requests[0].observed_ttft_ms is not None
    result = TraceReplayRunner(
        TraceReplayConfig(
            trace_path=str(trace_path),
            cache_token_budget=10000,
            policies=["lru", "adaptive_semantic"],
            output_dir=str(tmp_path / "replay"),
        )
    ).run(write_outputs=True)
    assert "lru" in result["metrics_by_policy"]

