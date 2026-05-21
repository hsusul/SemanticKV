from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256

from semantic_kv.cache.models import PromptBlock, SemanticType
from semantic_kv.routing.models import RoutingConfig, RoutingDecision
from semantic_kv.routing.replica_state import ReplicaRuntimeState, ReplicaStateStore


class RoutingPolicy(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def choose(
        self,
        request_id: str,
        blocks: list[PromptBlock],
        store: ReplicaStateStore,
        config: RoutingConfig,
        session_id: str | None = None,
    ) -> RoutingDecision:
        raise NotImplementedError


class RoundRobinRoutingPolicy(RoutingPolicy):
    def __init__(self) -> None:
        self._index = 0

    def name(self) -> str:
        return "round_robin"

    def choose(self, request_id: str, blocks: list[PromptBlock], store: ReplicaStateStore, config: RoutingConfig, session_id: str | None = None) -> RoutingDecision:
        states = store.healthy_states()
        if not states:
            raise ValueError("No healthy replicas available")
        state = states[self._index % len(states)]
        self._index += 1
        overlap, by_type = state.overlap(blocks)
        return RoutingDecision(
            request_id=request_id,
            selected_replica_id=state.replica.replica_id,
            reason="round-robin healthy replica selection",
            estimated_prefix_overlap_tokens=overlap,
            semantic_overlap_by_type=by_type,
            fallback_used=False,
        )


class LeastLoadedRoutingPolicy(RoutingPolicy):
    def name(self) -> str:
        return "least_loaded"

    def choose(self, request_id: str, blocks: list[PromptBlock], store: ReplicaStateStore, config: RoutingConfig, session_id: str | None = None) -> RoutingDecision:
        states = store.healthy_states()
        if not states:
            raise ValueError("No healthy replicas available")
        state = min(states, key=lambda s: (s.replica.current_load, s.replica.replica_id))
        overlap, by_type = state.overlap(blocks)
        return RoutingDecision(
            request_id=request_id,
            selected_replica_id=state.replica.replica_id,
            reason="least current load",
            estimated_prefix_overlap_tokens=overlap,
            semantic_overlap_by_type=by_type,
        )


class PrefixHashRoutingPolicy(RoutingPolicy):
    def name(self) -> str:
        return "prefix_hash"

    def choose(self, request_id: str, blocks: list[PromptBlock], store: ReplicaStateStore, config: RoutingConfig, session_id: str | None = None) -> RoutingDecision:
        states = store.healthy_states()
        if not states:
            raise ValueError("No healthy replicas available")
        key = "|".join(block.content_hash for block in blocks[:3]) or request_id
        index = int(sha256(key.encode("utf-8")).hexdigest(), 16) % len(states)
        state = sorted(states, key=lambda s: s.replica.replica_id)[index]
        overlap, by_type = state.overlap(blocks)
        return RoutingDecision(
            request_id=request_id,
            selected_replica_id=state.replica.replica_id,
            reason="stable hash of leading prompt block hashes",
            estimated_prefix_overlap_tokens=overlap,
            semantic_overlap_by_type=by_type,
        )


class SemanticLocalityRoutingPolicy(RoutingPolicy):
    def name(self) -> str:
        return "semantic_locality"

    def choose(self, request_id: str, blocks: list[PromptBlock], store: ReplicaStateStore, config: RoutingConfig, session_id: str | None = None) -> RoutingDecision:
        states = store.healthy_states()
        if not states:
            raise ValueError("No healthy replicas available")
        affinity_replica = store.session_affinity.get(session_id or "")
        scores: dict[str, float] = {}
        for state in states:
            score = state.semantic_overlap_score(blocks, config.semantic_weights)
            if config.session_affinity and affinity_replica == state.replica.replica_id:
                score += config.session_affinity_bonus
            score -= config.max_load_penalty * state.replica.current_load
            scores[state.replica.replica_id] = score
        if max(scores.values()) <= 0:
            cold_decision = PrefixHashRoutingPolicy().choose(request_id, blocks, store, config, session_id=session_id)
            cold_decision.reason = "cold-prefix fallback to stable prefix hash"
            cold_decision.score_by_replica = scores
            return cold_decision
        selected = max(
            sorted(states, key=lambda s: s.replica.replica_id),
            key=lambda s: (scores[s.replica.replica_id], -s.replica.current_load),
        )
        overlap, by_type = selected.overlap(blocks)
        return RoutingDecision(
            request_id=request_id,
            selected_replica_id=selected.replica.replica_id,
            reason="highest semantic cache-locality score",
            estimated_prefix_overlap_tokens=overlap,
            semantic_overlap_by_type=by_type,
            score_by_replica=scores,
        )


def build_routing_policy(name: str) -> RoutingPolicy:
    policies: dict[str, type[RoutingPolicy]] = {
        "round_robin": RoundRobinRoutingPolicy,
        "least_loaded": LeastLoadedRoutingPolicy,
        "prefix_hash": PrefixHashRoutingPolicy,
        "semantic_locality": SemanticLocalityRoutingPolicy,
    }
    try:
        return policies[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown routing policy: {name}") from exc
