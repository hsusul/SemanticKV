from __future__ import annotations

import os

from fastapi import FastAPI, Request

from semantic_kv.adapters.openai_proxy.models import OpenAIProxySettings
from semantic_kv.adapters.openai_proxy.proxy import OpenAICompatibleTelemetryProxy


def _settings_from_env() -> OpenAIProxySettings:
    return OpenAIProxySettings(
        upstream_base_url=os.getenv("SEMANTICKV_UPSTREAM_BASE_URL", "http://localhost:8001"),
        trace_output_dir=os.getenv("SEMANTICKV_TRACE_OUTPUT_DIR", "outputs/live_traces"),
        redact_text=os.getenv("SEMANTICKV_REDACT_TEXT", "true").lower() not in {"0", "false", "no"},
        timeout_seconds=float(os.getenv("SEMANTICKV_PROXY_TIMEOUT_SECONDS", "60")),
    )


app = FastAPI(title="SemanticKV OpenAI-Compatible Telemetry Proxy", version="0.1.0")
proxy = OpenAICompatibleTelemetryProxy(_settings_from_env())


@app.get("/health")
def health() -> dict[str, str | bool | float]:
    settings = proxy.settings
    return {
        "status": "ok",
        "service": "semantic-kv-openai-proxy",
        "upstream_base_url": settings.upstream_base_url,
        "redact_text": settings.redact_text,
        "timeout_seconds": settings.timeout_seconds,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy.forward_chat_completions(request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy.forward_completions(request)


@app.get("/v1/traces/latest")
def latest_trace():
    return proxy.latest_trace_summary()

