from __future__ import annotations

from semantic_kv.cache.models import CacheDecision
from semantic_kv.cache.store import CacheState
from semantic_kv.policies.base import EvictionPolicy


class FIFOPolicy(EvictionPolicy):
    def name(self) -> str:
        return "fifo"

    def choose_evictions(self, required_tokens: int, state: CacheState) -> CacheDecision:
        target = max(0, state.used_tokens + required_tokens - state.capacity_tokens)
        freed = 0
        evict: list[str] = []
        ordered = sorted(state.blocks.values(), key=lambda b: (b.created_at_ms, b.cache_key))
        for block in ordered:
            if freed >= target:
                break
            evict.append(block.cache_key)
            freed += block.token_count
        return self._decision(state, evict, {k: float(i) for i, k in enumerate(evict)}, "fifo oldest first")

