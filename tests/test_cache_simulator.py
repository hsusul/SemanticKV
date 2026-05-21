from semantic_kv.cache.models import PromptRequest
from semantic_kv.cache.simulator import PrefixCacheSimulator
from semantic_kv.policies.lru import LRUPolicy


def test_simulator_hits_and_capacity_invariant():
    simulator = PrefixCacheSimulator(policy=LRUPolicy(), capacity_tokens=30, token_latency_ms=1.0, base_ttft_ms=10.0)
    req = PromptRequest(request_id="r1", messages=[{"role": "system", "content": "shared system prompt"}])
    first = simulator.simulate_request(req)
    second = simulator.simulate_request(req.model_copy(update={"request_id": "r2"}))
    assert first.misses == 1
    assert second.hits == 1
    assert second.tokens_saved > 0
    assert simulator.state.used_tokens <= simulator.state.capacity_tokens


def test_simulator_rejects_oversized_block():
    simulator = PrefixCacheSimulator(policy=LRUPolicy(), capacity_tokens=3)
    req = PromptRequest(request_id="r1", raw_prompt=" ".join(["token"] * 20))
    result = simulator.simulate_request(req)
    assert result.rejected_blocks
    assert simulator.state.used_tokens <= 3

