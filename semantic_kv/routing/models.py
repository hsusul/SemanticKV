from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from semantic_kv.cache.models import SemanticType
from semantic_kv.policies.base import DEFAULT_SEMANTIC_WEIGHTS


class BackendReplica(BaseModel):
    replica_id: str
    base_url: str
    model_name: str | None = None
    healthy: bool = True
    current_load: float = 0.0
    estimated_cached_block_hashes: set[str] = Field(default_factory=set)
    last_seen_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    request_id: str
    selected_replica_id: str
    reason: str
    estimated_prefix_overlap_tokens: int = 0
    semantic_overlap_by_type: dict[SemanticType, int] = Field(default_factory=dict)
    fallback_used: bool = False
    score_by_replica: dict[str, float] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    replicas: list[BackendReplica]
    policy_name: str = "semantic_locality"
    max_load_penalty: float = 1.0
    semantic_weights: dict[SemanticType, float] = Field(default_factory=lambda: dict(DEFAULT_SEMANTIC_WEIGHTS))
    session_affinity: bool = True
    session_affinity_bonus: float = 500.0
    max_tracked_tokens_per_replica: int = 10000

