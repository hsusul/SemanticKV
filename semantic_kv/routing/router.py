from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from semantic_kv.adapters.openai_proxy.client import OpenAICompatibleClient
from semantic_kv.cache.models import PromptBlock, PromptRequest
from semantic_kv.collectors.jsonl_collector import JsonlTraceCollector
from semantic_kv.routing.models import BackendReplica, RoutingConfig, RoutingDecision
from semantic_kv.routing.policies import RoutingPolicy, build_routing_policy
from semantic_kv.routing.replica_state import ReplicaStateStore
from semantic_kv.segmentation.segmenter import PromptSegmenter
from semantic_kv.utils.time import now_ms


class SemanticKVRouter:
    def __init__(
        self,
        config: RoutingConfig,
        policy: RoutingPolicy | None = None,
        trace_output_dir: str = "outputs/live_traces",
        redact_text: bool = True,
        timeout_seconds: float = 60.0,
        transports_by_replica: dict[str, httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or build_routing_policy(config.policy_name)
        self.store = ReplicaStateStore(config.replicas, config.max_tracked_tokens_per_replica)
        self.segmenter = PromptSegmenter()
        self.trace_output_dir = trace_output_dir
        self.redact_text = redact_text
        self.timeout_seconds = timeout_seconds
        self.transports_by_replica = transports_by_replica or {}
        self.latest_trace_path: str | None = None
        self.decisions: list[RoutingDecision] = []
        self.fallback_count = 0

    def route_blocks(self, request_id: str, blocks: list[PromptBlock], session_id: str | None = None) -> RoutingDecision:
        decision = self.policy.choose(request_id, blocks, self.store, self.config, session_id=session_id)
        self.decisions.append(decision)
        return decision

    def update_replica_state(self, replica_id: str, blocks: list[PromptBlock], session_id: str | None = None) -> None:
        self.store.observe(replica_id, blocks, session_id=session_id)

    async def forward_chat(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> tuple[httpx.Response, RoutingDecision]:
        request_id = str(payload.get("request_id") or f"route_req_{now_ms()}")
        model_name = str(payload.get("model", "unknown-model"))
        blocks = self._segment(payload, request_id, session_id, tenant_id)
        decision = self.route_blocks(request_id, blocks, session_id=session_id)
        replica = self.store.get(decision.selected_replica_id).replica
        replica.current_load += 1
        started = now_ms()
        collector = self._collector()
        span = collector.start_request(
            request_id=request_id,
            messages=payload.get("messages") or [],
            model_name=model_name,
            session_id=session_id,
            tenant_id=tenant_id,
            metadata={
                "adapter": "semantic_router",
                "selected_replica_id": replica.replica_id,
                "routing_policy": self.policy.name(),
                "routing_reason": decision.reason,
                "estimated_prefix_overlap_tokens": decision.estimated_prefix_overlap_tokens,
                "semantic_overlap_by_type": {k.value: v for k, v in decision.semantic_overlap_by_type.items()},
                "backend_url": replica.base_url,
            },
        )
        try:
            client = OpenAICompatibleClient(
                replica.base_url,
                timeout_seconds=self.timeout_seconds,
                transport=self.transports_by_replica.get(replica.replica_id),
            )
            response = await client.post_json("/v1/chat/completions", payload, headers=headers)
            total_ms = float(now_ms() - started)
            span.set_backend_metadata(
                {
                    "status_code": response.status_code,
                    "observed_ttft_ms": total_ms,
                    "observed_total_latency_ms": total_ms,
                }
            )
            collector.finish_request(span)
            self.update_replica_state(replica.replica_id, blocks, session_id=session_id)
            return response, decision
        except BaseException as exc:
            collector.record_error(span, exc)
            raise
        finally:
            replica.current_load = max(0.0, replica.current_load - 1)
            collector.close()

    def replicas_snapshot(self) -> list[dict]:
        return self.store.snapshot()

    def latest_trace_summary(self) -> dict[str, Any]:
        if self.latest_trace_path is None:
            return {"trace_path": None, "exists": False, "requests": 0}
        path = Path(self.latest_trace_path)
        requests = 0
        if path.exists():
            requests = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return {"trace_path": self.latest_trace_path, "exists": path.exists(), "requests": requests}

    def _segment(self, payload: dict[str, Any], request_id: str, session_id: str | None, tenant_id: str | None) -> list[PromptBlock]:
        request = PromptRequest(
            request_id=request_id,
            session_id=session_id,
            tenant_id=tenant_id,
            timestamp_ms=now_ms(),
            messages=payload.get("messages") or [],
            model_name=payload.get("model"),
            metadata={"source": "semantic_router"},
        )
        return self.segmenter.segment(request)

    def _collector(self) -> JsonlTraceCollector:
        output_dir = Path(self.trace_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.latest_trace_path is None:
            self.latest_trace_path = str(output_dir / f"{now_ms()}_semantic_router_trace.jsonl")
        return JsonlTraceCollector(
            self.latest_trace_path,
            backend_name="semantic_router",
            drop_raw_text=self.redact_text,
            append=True,
        )


def replicas_from_urls(urls: list[str], model_name: str | None = None) -> list[BackendReplica]:
    return [
        BackendReplica(replica_id=f"replica-{idx + 1}", base_url=url.rstrip("/"), model_name=model_name)
        for idx, url in enumerate(urls)
    ]

