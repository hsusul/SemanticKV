from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from apps.router.main import app
import apps.router.main as router_main
from semantic_kv.cache.models import PromptBlock, SemanticType
from semantic_kv.routing.models import BackendReplica, RoutingConfig
from semantic_kv.routing.policies import (
    LeastLoadedRoutingPolicy,
    RoundRobinRoutingPolicy,
    SemanticLocalityRoutingPolicy,
)
from semantic_kv.routing.replica_state import ReplicaStateStore
from semantic_kv.routing.router import SemanticKVRouter


def block(name: str, typ: SemanticType = SemanticType.SYSTEM, tokens: int = 10) -> PromptBlock:
    return PromptBlock(
        block_id=name,
        request_id="r",
        position=0,
        text=name,
        token_count=tokens,
        semantic_type=typ,
        content_hash=f"h-{name}",
    )


def replicas() -> list[BackendReplica]:
    return [
        BackendReplica(replica_id="a", base_url="http://a", current_load=2),
        BackendReplica(replica_id="b", base_url="http://b", current_load=0),
    ]


def test_round_robin_policy_cycles_replicas():
    store = ReplicaStateStore(replicas())
    config = RoutingConfig(replicas=replicas(), policy_name="round_robin")
    policy = RoundRobinRoutingPolicy()
    first = policy.choose("r1", [], store, config)
    second = policy.choose("r2", [], store, config)
    third = policy.choose("r3", [], store, config)
    assert [first.selected_replica_id, second.selected_replica_id, third.selected_replica_id] == ["a", "b", "a"]


def test_least_loaded_chooses_lowest_load():
    store = ReplicaStateStore(replicas())
    config = RoutingConfig(replicas=replicas())
    decision = LeastLoadedRoutingPolicy().choose("r1", [], store, config)
    assert decision.selected_replica_id == "b"


def test_semantic_locality_prefers_matching_high_value_blocks():
    reps = replicas()
    store = ReplicaStateStore(reps)
    sys_block = block("shared-system", SemanticType.SYSTEM, 100)
    user_block = block("unique-user", SemanticType.USER, 100)
    store.observe("a", [sys_block])
    store.observe("b", [user_block])
    config = RoutingConfig(replicas=reps, max_load_penalty=0.0)
    decision = SemanticLocalityRoutingPolicy().choose("r1", [sys_block, user_block], store, config)
    assert decision.selected_replica_id == "a"
    assert decision.estimated_prefix_overlap_tokens == 100


def test_session_affinity_boosts_previous_replica():
    reps = replicas()
    store = ReplicaStateStore(reps)
    store.observe("b", [block("previous")], session_id="s1")
    config = RoutingConfig(replicas=reps, session_affinity=True, session_affinity_bonus=1000)
    decision = SemanticLocalityRoutingPolicy().choose("r1", [], store, config, session_id="s1")
    assert decision.selected_replica_id == "b"


def test_replica_state_lru_eviction_when_budget_exceeded():
    reps = [BackendReplica(replica_id="a", base_url="http://a")]
    store = ReplicaStateStore(reps, max_tracked_tokens_per_replica=15)
    store.observe("a", [block("one", tokens=10), block("two", tokens=10)])
    state = store.get("a")
    assert state.tracked_tokens <= 15
    assert "h-one" not in state.tracked_blocks
    assert "h-two" in state.tracked_blocks


def test_router_app_imports():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "semantic-kv-router"


def test_router_forwards_to_mock_backend(tmp_path: Path):
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"id": "router-test", "model": payload["model"], "choices": [{"message": {"content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    config = RoutingConfig(
        replicas=[BackendReplica(replica_id="a", base_url="http://a")],
        policy_name="semantic_locality",
    )
    router_main.router = SemanticKVRouter(
        config,
        trace_output_dir=str(tmp_path),
        transports_by_replica={"a": transport},
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        headers={"X-Session-ID": "s1"},
        json={"model": "mock", "messages": [{"role": "user", "content": "USER: hello"}]},
    )
    assert response.status_code == 200
    assert response.headers["X-SemanticKV-Replica"] == "a"
    assert list(tmp_path.glob("*_semantic_router_trace.jsonl"))

