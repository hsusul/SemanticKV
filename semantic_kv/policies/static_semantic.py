from __future__ import annotations

import math

from semantic_kv.cache.models import CacheBlock, CacheDecision, PromptBlock
from semantic_kv.cache.store import CacheState
from semantic_kv.policies.base import DEFAULT_SEMANTIC_WEIGHTS, EvictionPolicy


class StaticSemanticPolicy(EvictionPolicy):
    def name(self) -> str:
        return "static_semantic"

    def score(self, block: CacheBlock, now_ms: int | None = None) -> float:
        weight = DEFAULT_SEMANTIC_WEIGHTS[block.semantic_type]
        freq = math.log1p(block.access_count)
        latency = max(1.0, block.estimated_latency_saved_ms)
        return (weight * max(1.0, freq) * latency) / max(1.0, math.sqrt(block.token_count))

    def choose_evictions(self, required_tokens: int, state: CacheState) -> CacheDecision:
        target = max(0, state.used_tokens + required_tokens - state.capacity_tokens)
        freed = 0
        evict: list[str] = []
        scores = {block.cache_key: self.score(block) for block in state.blocks.values()}
        ordered = sorted(state.blocks.values(), key=lambda b: (scores[b.cache_key], b.last_accessed_at_ms, b.cache_key))
        for block in ordered:
            if freed >= target:
                break
            evict.append(block.cache_key)
            freed += block.token_count
        return self._decision(state, evict, scores, "lowest static semantic score")

    def should_admit(self, block: PromptBlock, state: CacheState) -> bool:
        if state.used_tokens + block.token_count <= state.capacity_tokens or not state.blocks:
            return True
        hypothetical = CacheBlock(
            cache_key=block.block_id,
            content_hash=block.content_hash,
            semantic_type=block.semantic_type,
            token_count=block.token_count,
            memory_bytes_estimate=block.token_count,
            created_at_ms=0,
            last_accessed_at_ms=0,
            estimated_latency_saved_ms=block.token_count,
        )
        incoming_score = self.score(hypothetical)
        retained_scores = [self.score(existing) for existing in state.blocks.values()]
        return incoming_score >= min(retained_scores)
