from __future__ import annotations

from pydantic import BaseModel, Field


class RequestSimulationResult(BaseModel):
    request_id: str
    policy_name: str
    hits: int
    misses: int
    hit_keys: list[str] = Field(default_factory=list)
    missed_hashes: list[str] = Field(default_factory=list)
    tokens_saved: int
    estimated_ttft_ms: float
    admitted_blocks: list[str] = Field(default_factory=list)
    evicted_blocks: list[str] = Field(default_factory=list)
    rejected_blocks: list[str] = Field(default_factory=list)
    decision_id: str | None = None

