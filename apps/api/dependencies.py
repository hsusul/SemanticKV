from __future__ import annotations

from functools import lru_cache

from semantic_kv.cache.simulator import PrefixCacheSimulator
from semantic_kv.policies.adaptive_semantic import AdaptiveSemanticPolicy


@lru_cache(maxsize=1)
def get_simulator() -> PrefixCacheSimulator:
    return PrefixCacheSimulator(policy=AdaptiveSemanticPolicy(), capacity_tokens=4000)

