from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import Response

from semantic_kv.routing.models import RoutingConfig
from semantic_kv.routing.router import SemanticKVRouter, replicas_from_urls


def _router_from_env() -> SemanticKVRouter:
    urls = [
        url.strip()
        for url in os.getenv("SEMANTICKV_REPLICAS", "http://localhost:8101,http://localhost:8102").split(",")
        if url.strip()
    ]
    config = RoutingConfig(
        replicas=replicas_from_urls(urls),
        policy_name=os.getenv("SEMANTICKV_ROUTING_POLICY", "semantic_locality"),
        session_affinity=os.getenv("SEMANTICKV_SESSION_AFFINITY", "true").lower() not in {"0", "false", "no"},
        max_tracked_tokens_per_replica=int(os.getenv("SEMANTICKV_MAX_TRACKED_TOKENS", "10000")),
    )
    return SemanticKVRouter(
        config,
        trace_output_dir=os.getenv("SEMANTICKV_TRACE_OUTPUT_DIR", "outputs/live_traces"),
        redact_text=os.getenv("SEMANTICKV_REDACT_TEXT", "true").lower() not in {"0", "false", "no"},
        timeout_seconds=float(os.getenv("SEMANTICKV_PROXY_TIMEOUT_SECONDS", "60")),
    )


app = FastAPI(title="SemanticKV Semantic Router", version="0.1.0")
router = _router_from_env()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "semantic-kv-router",
        "policy": router.policy.name(),
        "replicas": len(router.config.replicas),
    }


@app.get("/v1/replicas")
def replicas():
    return {"replicas": router.replicas_snapshot()}


@app.get("/v1/traces/latest")
def latest_trace():
    return router.latest_trace_summary()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    response, decision = await router.forward_chat(
        payload,
        headers={key: value for key, value in request.headers.items() if key.lower() not in {"host", "content-length"}},
        session_id=request.headers.get("X-Session-ID"),
        tenant_id=request.headers.get("X-Tenant-ID"),
    )
    headers = {"X-SemanticKV-Replica": decision.selected_replica_id}
    content_type = response.headers.get("content-type", "application/json")
    return Response(content=response.content, status_code=response.status_code, media_type=content_type, headers=headers)

