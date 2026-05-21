from __future__ import annotations

import math

from semantic_kv.cache.models import CacheBlock, CacheDecision, CacheEvent, PromptBlock, SemanticType
from semantic_kv.cache.store import CacheState
from semantic_kv.policies.base import DEFAULT_SEMANTIC_WEIGHTS, EvictionPolicy


class AdaptiveSemanticPolicy(EvictionPolicy):
    def __init__(
        self,
        recency_tau_ms: float = 60_000.0,
        size_penalty: float = 0.15,
        min_weight: float = 0.2,
        max_weight: float = 5.0,
        regret_horizon_events: int = 500,
    ) -> None:
        self.semantic_weights: dict[SemanticType, float] = dict(DEFAULT_SEMANTIC_WEIGHTS)
        self.recency_tau_ms = recency_tau_ms
        self.size_penalty = size_penalty
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.regret_horizon_events = regret_horizon_events

    def name(self) -> str:
        return "adaptive_semantic"

    def score(self, block: CacheBlock, now_ms: int | None = None) -> float:
        now = now_ms if now_ms is not None else block.last_accessed_at_ms
        age_ms = max(0, now - block.last_accessed_at_ms)
        recency_factor = math.exp(-age_ms / self.recency_tau_ms)
        score = (
            self.semantic_weights[block.semantic_type]
            * max(1.0, 1.0 + math.log1p(block.access_count) + (0.25 * block.access_count))
            * max(0.05, recency_factor)
            * max(1.0, block.estimated_latency_saved_ms)
            / max(1.0, block.token_count**self.size_penalty)
        )
        return score

    def choose_evictions(self, required_tokens: int, state: CacheState) -> CacheDecision:
        target = max(0, state.used_tokens + required_tokens - state.capacity_tokens)
        freed = 0
        evict: list[str] = []
        now = state.events[-1].timestamp_ms if state.events else 0
        scores = {block.cache_key: self.score(block, now) for block in state.blocks.values()}
        ordered = sorted(state.blocks.values(), key=lambda b: (scores[b.cache_key], b.last_accessed_at_ms, b.cache_key))
        for block in ordered:
            if freed >= target:
                break
            evict.append(block.cache_key)
            freed += block.token_count
        return self._decision(state, evict, scores, "lowest adaptive semantic score")

    def update_from_feedback(self, feedback: list[CacheEvent], state: CacheState) -> None:
        for event in feedback:
            if event.semantic_type is None:
                continue
            if event.event_type == "regret":
                self._bump(event.semantic_type, 0.18 + min(0.12, event.token_count / 1000.0))
            elif event.event_type == "hit":
                self._bump(event.semantic_type, 0.01 + min(0.04, event.token_count / 4000.0))
        self._decay_unused(state)

    def _bump(self, semantic_type: SemanticType, delta: float) -> None:
        self.semantic_weights[semantic_type] = min(self.max_weight, self.semantic_weights[semantic_type] + delta)

    def _decay_unused(self, state: CacheState) -> None:
        if not state.events:
            return
        now = state.events[-1].timestamp_ms
        for block in state.blocks.values():
            if block.access_count == 0 and now - block.created_at_ms > self.recency_tau_ms:
                self.semantic_weights[block.semantic_type] = max(
                    self.min_weight,
                    self.semantic_weights[block.semantic_type] * 0.995,
                )

    def export_state(self) -> dict[str, float]:
        return {k.value: round(v, 4) for k, v in self.semantic_weights.items()}

    def should_admit(self, block: PromptBlock, state: CacheState) -> bool:
        if state.used_tokens + block.token_count <= state.capacity_tokens or not state.blocks:
            return True
        if block.semantic_type in {SemanticType.USER, SemanticType.OTHER}:
            return False
        now = state.events[-1].timestamp_ms if state.events else 0
        hypothetical = CacheBlock(
            cache_key=block.block_id,
            content_hash=block.content_hash,
            semantic_type=block.semantic_type,
            token_count=block.token_count,
            memory_bytes_estimate=block.token_count,
            created_at_ms=now,
            last_accessed_at_ms=now,
            access_count=0,
            estimated_latency_saved_ms=block.token_count,
        )
        incoming_score = self.score(hypothetical, now)
        retained_scores = [self.score(existing, now) for existing in state.blocks.values()]
        return incoming_score >= min(retained_scores)
