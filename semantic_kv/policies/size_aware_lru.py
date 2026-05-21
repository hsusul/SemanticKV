from __future__ import annotations

from semantic_kv.cache.models import CacheDecision
from semantic_kv.cache.store import CacheState
from semantic_kv.policies.base import EvictionPolicy


class SizeAwareLRUPolicy(EvictionPolicy):
    def name(self) -> str:
        return "size_aware_lru"

    def choose_evictions(self, required_tokens: int, state: CacheState) -> CacheDecision:
        target = max(0, state.used_tokens + required_tokens - state.capacity_tokens)
        freed = 0
        evict: list[str] = []
        scores: dict[str, float] = {}
        ordered = sorted(
            state.blocks.values(),
            key=lambda b: ((b.last_accessed_at_ms / max(1, b.token_count)), b.last_accessed_at_ms, b.cache_key),
        )
        for block in ordered:
            if freed >= target:
                break
            evict.append(block.cache_key)
            scores[block.cache_key] = block.last_accessed_at_ms / max(1, block.token_count)
            freed += block.token_count
        return self._decision(state, evict, scores, "size-aware least recently used")

