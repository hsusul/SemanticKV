from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from semantic_kv.cache.models import SemanticType


class TracePromptBlock(BaseModel):
    block_id: str | None = None
    content_hash: str | None = None
    semantic_type: SemanticType
    token_count: int
    text: str | None = None
    position: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRequest(BaseModel):
    request_id: str
    timestamp_ms: int
    session_id: str | None = None
    tenant_id: str | None = None
    model_name: str | None = None
    blocks: list[TracePromptBlock]
    observed_ttft_ms: float | None = None
    observed_total_latency_ms: float | None = None
    observed_prefix_hit_tokens: int | None = None
    observed_prefill_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(block.token_count for block in self.blocks)


class ServingTrace(BaseModel):
    trace_id: str
    backend_name: str = "synthetic"
    model_name: str | None = None
    requests: list[TraceRequest]
    metadata: dict[str, Any] = Field(default_factory=dict)


class LatencyModelConfig(BaseModel):
    kind: str = "linear"
    base_ms: float = 80.0
    ms_per_uncached_token: float = 0.08
    long_context_threshold_tokens: int | None = None
    long_context_penalty_ms_per_token: float = 0.0


class TraceReplayConfig(BaseModel):
    trace_path: str
    cache_token_budget: int
    policies: list[str] = Field(default_factory=lambda: ["lru", "adaptive_semantic", "static_semantic"])
    latency_model: LatencyModelConfig = Field(default_factory=LatencyModelConfig)
    output_name: str = "trace_replay"
    random_seed: int | None = None
    calibrate_from_trace: bool = False
    output_dir: str = "outputs/traces"

