from semantic_kv.cache.models import CacheBlock, CacheEvent, SemanticType
from semantic_kv.cache.store import CacheState
from semantic_kv.policies.adaptive_semantic import AdaptiveSemanticPolicy
from semantic_kv.policies.lfu import LFUPolicy
from semantic_kv.policies.lru import LRUPolicy


def block(key: str, tokens: int, last: int, count: int, typ: SemanticType = SemanticType.USER) -> CacheBlock:
    return CacheBlock(
        cache_key=key,
        content_hash=key,
        semantic_type=typ,
        token_count=tokens,
        memory_bytes_estimate=tokens * 10,
        created_at_ms=last,
        last_accessed_at_ms=last,
        access_count=count,
        estimated_latency_saved_ms=tokens,
    )


def test_lru_evicts_oldest_access():
    state = CacheState(capacity_tokens=10)
    state.add(block("old", 5, 1, 3))
    state.add(block("new", 5, 10, 0))
    decision = LRUPolicy().choose_evictions(5, state)
    assert decision.evict_keys == ["old"]


def test_lfu_evicts_lowest_frequency():
    state = CacheState(capacity_tokens=10)
    state.add(block("rare", 5, 10, 0))
    state.add(block("freq", 5, 1, 4))
    decision = LFUPolicy().choose_evictions(5, state)
    assert decision.evict_keys == ["rare"]


def test_adaptive_score_prefers_system_over_user():
    policy = AdaptiveSemanticPolicy()
    system = block("sys", 10, 10, 3, SemanticType.SYSTEM)
    user = block("usr", 10, 10, 3, SemanticType.USER)
    assert policy.score(system, 10) > policy.score(user, 10)


def test_adaptive_weight_update_on_regret():
    policy = AdaptiveSemanticPolicy()
    state = CacheState(capacity_tokens=10)
    before = policy.semantic_weights[SemanticType.RETRIEVED_CONTEXT]
    event = CacheEvent(
        event_id="e",
        timestamp_ms=1,
        event_type="regret",
        semantic_type=SemanticType.RETRIEVED_CONTEXT,
        token_count=100,
        policy_name=policy.name(),
    )
    policy.update_from_feedback([event], state)
    assert policy.semantic_weights[SemanticType.RETRIEVED_CONTEXT] > before

