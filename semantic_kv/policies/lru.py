from __future__ import annotations

from semantic_kv.cache.models import CacheDecision
from semantic_kv.cache.store import CacheState
from semantic_kv.policies.base import EvictionPolicy


class LRUPolicy(EvictionPolicy):
    def name(self) -> str:
        return "lru"

    def choose_evictions(self, required_tokens: int, state: CacheState) -> CacheDecision:
        target = max(0, state.used_tokens + required_tokens - state.capacity_tokens)
        freed = 0
        evict: list[str] = []
        scores: dict[str, float] = {}
        ordered = sorted(state.blocks.values(), key=lambda b: (b.last_accessed_at_ms, b.created_at_ms, b.cache_key))
        for block in ordered:
            if freed >= target:
                break
            evict.append(block.cache_key)
            scores[block.cache_key] = float(block.last_accessed_at_ms)
            freed += block.token_count
        return self._decision(state, evict, scores, "least recently used")

