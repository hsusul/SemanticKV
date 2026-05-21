from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from semantic_kv.cache.models import PromptBlock, SemanticType
from semantic_kv.routing.models import BackendReplica
from semantic_kv.utils.time import now_ms


@dataclass
class TrackedBlock:
    content_hash: str
    token_count: int
    semantic_type: SemanticType
    last_seen_ms: int


@dataclass
class ReplicaRuntimeState:
    replica: BackendReplica
    max_tracked_tokens: int = 10000
    tracked_blocks: OrderedDict[str, TrackedBlock] = field(default_factory=OrderedDict)

    @property
    def tracked_tokens(self) -> int:
        return sum(block.token_count for block in self.tracked_blocks.values())

    def observe_blocks(self, blocks: list[PromptBlock], timestamp_ms: int | None = None) -> None:
        seen_at = timestamp_ms or now_ms()
        for block in blocks:
            existing = self.tracked_blocks.pop(block.content_hash, None)
            token_count = existing.token_count if existing else block.token_count
            self.tracked_blocks[block.content_hash] = TrackedBlock(
                content_hash=block.content_hash,
                token_count=token_count,
                semantic_type=block.semantic_type,
                last_seen_ms=seen_at,
            )
        self._evict_until_within_budget()
        self.replica.estimated_cached_block_hashes = set(self.tracked_blocks)
        self.replica.last_seen_ms = seen_at

    def overlap(self, blocks: list[PromptBlock]) -> tuple[int, dict[SemanticType, int]]:
        total = 0
        by_type: dict[SemanticType, int] = {}
        for block in blocks:
            if block.content_hash in self.tracked_blocks:
                total += block.token_count
                by_type[block.semantic_type] = by_type.get(block.semantic_type, 0) + block.token_count
        return total, by_type

    def semantic_overlap_score(self, blocks: list[PromptBlock], weights: dict[SemanticType, float]) -> float:
        score = 0.0
        for block in blocks:
            if block.content_hash in self.tracked_blocks:
                score += weights.get(block.semantic_type, 1.0) * block.token_count
        return score

    def _evict_until_within_budget(self) -> None:
        while self.tracked_tokens > self.max_tracked_tokens and self.tracked_blocks:
            self.tracked_blocks.popitem(last=False)


class ReplicaStateStore:
    def __init__(self, replicas: list[BackendReplica], max_tracked_tokens_per_replica: int = 10000) -> None:
        self.states: dict[str, ReplicaRuntimeState] = {
            replica.replica_id: ReplicaRuntimeState(replica, max_tracked_tokens_per_replica)
            for replica in replicas
        }
        self.session_affinity: dict[str, str] = {}

    def healthy_states(self) -> list[ReplicaRuntimeState]:
        return [state for state in self.states.values() if state.replica.healthy]

    def get(self, replica_id: str) -> ReplicaRuntimeState:
        return self.states[replica_id]

    def observe(self, replica_id: str, blocks: list[PromptBlock], session_id: str | None = None) -> None:
        self.states[replica_id].observe_blocks(blocks)
        if session_id:
            self.session_affinity[session_id] = replica_id

    def snapshot(self) -> list[dict]:
        return [
            {
                **state.replica.model_dump(mode="json"),
                "tracked_tokens": state.tracked_tokens,
                "tracked_block_count": len(state.tracked_blocks),
            }
            for state in self.states.values()
        ]

