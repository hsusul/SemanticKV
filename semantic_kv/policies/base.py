from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from semantic_kv.cache.models import CacheBlock, CacheDecision, CacheEvent, PromptBlock, SemanticType
from semantic_kv.cache.store import CacheState
from semantic_kv.utils.hashing import stable_hash


DEFAULT_SEMANTIC_WEIGHTS: dict[SemanticType, float] = {
    SemanticType.SYSTEM: 2.4,
    SemanticType.TEMPLATE: 2.0,
    SemanticType.RETRIEVED_CONTEXT: 1.4,
    SemanticType.TOOL_OUTPUT: 1.1,
    SemanticType.ASSISTANT: 1.0,
    SemanticType.CODE: 1.6,
    SemanticType.USER: 0.7,
    SemanticType.OTHER: 0.8,
}


class EvictionPolicy(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def on_access(self, block: CacheBlock, event: CacheEvent, state: CacheState) -> None:
        return None

    def on_admit(self, block: CacheBlock, state: CacheState) -> None:
        return None

    @abstractmethod
    def choose_evictions(self, required_tokens: int, state: CacheState) -> CacheDecision:
        raise NotImplementedError

    def update_from_feedback(self, feedback: list[CacheEvent], state: CacheState) -> None:
        return None

    def should_admit(self, block: PromptBlock, state: CacheState) -> bool:
        return True

    def _decision(self, state: CacheState, evict_keys: list[str], scores: dict[str, float], reason: str) -> CacheDecision:
        after = state.used_tokens - sum(state.blocks[k].token_count for k in evict_keys if k in state.blocks)
        return CacheDecision(
            decision_id=f"dec_{stable_hash(self.name(), len(state.events), state.used_tokens)}",
            policy_name=self.name(),
            evict_keys=evict_keys,
            score_by_key=scores,
            memory_before_tokens=state.used_tokens,
            memory_after_tokens=after,
            reason=reason,
        )

    def export_state(self) -> dict[str, Any]:
        return {}


def build_policy(name: str) -> EvictionPolicy:
    from semantic_kv.policies.adaptive_semantic import AdaptiveSemanticPolicy
    from semantic_kv.policies.fifo import FIFOPolicy
    from semantic_kv.policies.lfu import LFUPolicy
    from semantic_kv.policies.lru import LRUPolicy
    from semantic_kv.policies.size_aware_lru import SizeAwareLRUPolicy
    from semantic_kv.policies.static_semantic import StaticSemanticPolicy

    policies: dict[str, type[EvictionPolicy]] = {
        "fifo": FIFOPolicy,
        "lru": LRUPolicy,
        "lfu": LFUPolicy,
        "size_aware_lru": SizeAwareLRUPolicy,
        "static_semantic": StaticSemanticPolicy,
        "adaptive_semantic": AdaptiveSemanticPolicy,
    }
    try:
        return policies[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown policy: {name}") from exc
