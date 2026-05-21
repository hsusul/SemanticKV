from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SemanticType(StrEnum):
    SYSTEM = "system"
    USER = "user"
    RETRIEVED_CONTEXT = "retrieved_context"
    TOOL_OUTPUT = "tool_output"
    ASSISTANT = "assistant"
    TEMPLATE = "template"
    CODE = "code"
    OTHER = "other"


class PromptRequest(BaseModel):
    request_id: str
    session_id: str | None = None
    tenant_id: str | None = None
    timestamp_ms: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    raw_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int | None = None
    model_name: str | None = None


class PromptBlock(BaseModel):
    block_id: str
    request_id: str
    session_id: str | None = None
    tenant_id: str | None = None
    position: int
    text: str
    token_count: int
    semantic_type: SemanticType
    source: str | None = None
    content_hash: str
    is_private: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheBlock(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cache_key: str
    content_hash: str
    semantic_type: SemanticType
    token_count: int
    memory_bytes_estimate: int
    created_at_ms: int
    last_accessed_at_ms: int
    access_count: int = 0
    session_ids: set[str] = Field(default_factory=set)
    tenant_id: str | None = None
    score: float = 0.0
    estimated_latency_saved_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheEvent(BaseModel):
    event_id: str
    timestamp_ms: int
    request_id: str | None = None
    event_type: Literal["hit", "miss", "admit", "evict", "reject", "regret"]
    cache_key: str | None = None
    content_hash: str | None = None
    semantic_type: SemanticType | None = None
    token_count: int = 0
    policy_name: str
    reason: str | None = None
    estimated_latency_delta_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheDecision(BaseModel):
    decision_id: str
    request_id: str | None = None
    policy_name: str
    admit_keys: list[str] = Field(default_factory=list)
    evict_keys: list[str] = Field(default_factory=list)
    reject_keys: list[str] = Field(default_factory=list)
    score_by_key: dict[str, float] = Field(default_factory=dict)
    score_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)
    memory_before_tokens: int = 0
    memory_after_tokens: int = 0
    reason: str = ""


class PolicyMetrics(BaseModel):
    policy_name: str
    requests_total: int = 0
    hits_total: int = 0
    misses_total: int = 0
    evictions_total: int = 0
    hit_rate: float = 0.0
    tokens_saved_total: int = 0
    estimated_ttft_ms_avg: float = 0.0
    estimated_ttft_ms_p50: float = 0.0
    estimated_ttft_ms_p95: float = 0.0
    memory_tokens_used_avg: float = 0.0
    memory_utilization_avg: float = 0.0
    eviction_regret_total: float = 0.0
    relative_ttft_improvement_vs_lru: float | None = None
    relative_hit_rate_improvement_vs_lru: float | None = None
    hit_rate_by_type: dict[SemanticType, float] = Field(default_factory=dict)
    evictions_by_type: dict[SemanticType, int] = Field(default_factory=dict)
    final_semantic_weights: dict[str, float] = Field(default_factory=dict)


class WorkloadSession(BaseModel):
    session_id: str
    workload_type: str
    tenant_id: str | None = None
    start_timestamp_ms: int = 0
    requests: list[PromptRequest] = Field(default_factory=list)
    shared_artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    experiment_id: str = "default"
    seed: int = 7
    policies: list[str] = Field(
        default_factory=lambda: [
            "fifo",
            "lru",
            "lfu",
            "size_aware_lru",
            "static_semantic",
            "adaptive_semantic",
        ]
    )
    workload_families: list[str] = Field(default_factory=lambda: ["repeated_system", "rag_shared_corpus"])
    num_sessions: int = 50
    requests_per_session: int = 6
    cache_capacity_tokens: int = 4000
    token_latency_ms: float = 0.08
    base_ttft_ms: float = 80.0
    semantic_classifier: str = "rule_based"
    output_dir: str = "outputs/experiments"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    experiment_id: str
    config: ExperimentConfig
    metrics_by_policy: dict[str, PolicyMetrics]
    best_policy_by_ttft: str
    best_policy_by_hit_rate: str
    output_dir: str
    timeseries_by_policy: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    summary: str = ""
