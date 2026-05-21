from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from apps.api.dependencies import get_simulator
from semantic_kv.collectors.jsonl_collector import JsonlTraceCollector
from semantic_kv.collectors.mock_backend import MockLLMBackend
from semantic_kv.cache.models import ExperimentConfig, PromptRequest
from semantic_kv.cache.simulator import PrefixCacheSimulator
from semantic_kv.experiments.runner import ExperimentRunner
from semantic_kv.utils.time import now_ms

app = FastAPI(title="SemanticKV API", version="0.1.0")
_mock_backend = MockLLMBackend()
_latest_trace_path: str | None = None


class MockGenerateRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str = "mock-llm"
    session_id: str | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    drop_raw_text: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "semantic-kv-api", "version": "0.1.0"}


@app.post("/v1/simulate/request")
def simulate_request(request: PromptRequest, simulator: PrefixCacheSimulator = Depends(get_simulator)):
    return simulator.simulate_request(request)


@app.post("/v1/experiments/run")
def run_experiment(config: ExperimentConfig):
    result = ExperimentRunner(config).run(write_outputs=True)
    return {
        "experiment_id": result.experiment_id,
        "best_policy_by_ttft": result.best_policy_by_ttft,
        "best_policy_by_hit_rate": result.best_policy_by_hit_rate,
        "output_dir": result.output_dir,
        "metrics_by_policy": result.metrics_by_policy,
    }


@app.get("/v1/cache/state")
def cache_state(simulator: PrefixCacheSimulator = Depends(get_simulator)):
    return simulator.state.snapshot()


@app.get("/v1/metrics/summary")
def metrics_summary(simulator: PrefixCacheSimulator = Depends(get_simulator)):
    return simulator.summary()


@app.post("/v1/mock/generate")
def mock_generate(request: MockGenerateRequest):
    global _latest_trace_path
    output_dir = Path("outputs/live_traces")
    output_dir.mkdir(parents=True, exist_ok=True)
    if _latest_trace_path is None:
        _latest_trace_path = str(output_dir / f"{now_ms()}_mock_backend_trace.jsonl")
    prompt = request.prompt or _messages_to_prompt(request.messages)
    collector = JsonlTraceCollector(
        output_path=_latest_trace_path,
        backend_name="mock_backend",
        model_name=request.model_name,
        drop_raw_text=request.drop_raw_text,
        append=True,
    )
    try:
        with collector.trace_request(
            prompt=prompt,
            messages=request.messages,
            model_name=request.model_name,
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            metadata={"route": "/v1/mock/generate", **request.metadata},
        ) as span:
            response = _mock_backend.generate(prompt)
            span.set_response_text(response["text"])
            span.set_observed_output_tokens(response["output_tokens"])
            span.set_observed_prefix_hit_tokens(response["observed_prefix_hit_tokens"])
            span.set_observed_prefill_tokens(response["observed_prefill_tokens"])
            span.set_backend_metadata(response["metadata"])
    finally:
        collector.close()
    return {"response": response, "trace_path": _latest_trace_path}


@app.get("/v1/traces/latest")
def latest_trace():
    if _latest_trace_path is None:
        return {"trace_path": None, "exists": False, "requests": 0}
    path = Path(_latest_trace_path)
    requests = 0
    if path.exists():
        requests = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {"trace_path": _latest_trace_path, "exists": path.exists(), "requests": requests}


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"{message.get('role', 'user').upper()}: {message.get('content', '')}" for message in messages)
