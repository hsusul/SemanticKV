from __future__ import annotations

from semantic_kv.cache.models import CacheBlock, CacheDecision, CacheEvent, PromptBlock, PromptRequest
from semantic_kv.cache.store import CacheState
from semantic_kv.metrics.collector import MetricsCollector
from semantic_kv.metrics.schemas import RequestSimulationResult
from semantic_kv.policies.base import EvictionPolicy
from semantic_kv.segmentation.segmenter import PromptSegmenter
from semantic_kv.utils.hashing import stable_hash


class PrefixCacheSimulator:
    def __init__(
        self,
        policy: EvictionPolicy,
        capacity_tokens: int = 4000,
        token_latency_ms: float = 0.08,
        base_ttft_ms: float = 80.0,
        segmenter: PromptSegmenter | None = None,
        bytes_per_token: int = 4096,
    ) -> None:
        self.policy = policy
        self.state = CacheState(capacity_tokens=capacity_tokens)
        self.token_latency_ms = token_latency_ms
        self.base_ttft_ms = base_ttft_ms
        self.bytes_per_token = bytes_per_token
        self.segmenter = segmenter or PromptSegmenter()
        self.metrics = MetricsCollector(policy.name(), capacity_tokens)
        self._clock_ms = 1

    def simulate_request(self, request: PromptRequest) -> RequestSimulationResult:
        blocks = self.segmenter.segment(request)
        return self.process_blocks(request, blocks)

    def process_blocks(self, request: PromptRequest, blocks: list[PromptBlock]) -> RequestSimulationResult:
        self._clock_ms = max(self._clock_ms + 1, request.timestamp_ms or 0)
        request_events: list[CacheEvent] = []
        hits = 0
        misses = 0
        tokens_saved = 0
        hit_keys: list[str] = []
        missed_blocks: list[PromptBlock] = []
        missed_hashes: list[str] = []

        for block in blocks:
            cached = self.state.get_by_hash(block.content_hash)
            if cached is not None:
                hits += 1
                cached.access_count += 1
                cached.last_accessed_at_ms = self._clock_ms
                if request.session_id:
                    cached.session_ids.add(request.session_id)
                cached.estimated_latency_saved_ms += block.token_count * self.token_latency_ms
                tokens_saved += block.token_count
                hit_keys.append(cached.cache_key)
                event = self._event("hit", request, block, cached.cache_key, block.token_count * self.token_latency_ms)
                self.policy.on_access(cached, event, self.state)
                request_events.append(event)
                self.state.append_event(event)
            else:
                misses += 1
                missed_blocks.append(block)
                missed_hashes.append(block.content_hash)
                event = self._event("miss", request, block, None, block.token_count * self.token_latency_ms)
                request_events.append(event)
                self.state.append_event(event)
                if block.content_hash in self.state.evicted_recent:
                    previous = self.state.evicted_recent[block.content_hash]
                    regret = self._event(
                        "regret",
                        request,
                        block,
                        previous.cache_key,
                        block.token_count * self.token_latency_ms,
                        reason="requested after eviction",
                    )
                    request_events.append(regret)
                    self.state.append_event(regret)

        admitted: list[str] = []
        rejected: list[str] = []
        evicted: list[str] = []
        decision: CacheDecision | None = None

        for block in missed_blocks:
            if block.token_count > self.state.capacity_tokens:
                rejected.append(block.block_id)
                event = self._event("reject", request, block, None, 0.0, reason="block exceeds capacity")
                request_events.append(event)
                self.state.append_event(event)
                continue
            if not self.policy.should_admit(block, self.state):
                rejected.append(block.block_id)
                event = self._event("reject", request, block, None, 0.0, reason="admission score below retained cache")
                request_events.append(event)
                self.state.append_event(event)
                continue
            decision = self.policy.choose_evictions(block.token_count, self.state)
            for key in decision.evict_keys:
                removed = self.state.remove(key)
                if removed is None:
                    continue
                evicted.append(key)
                event = CacheEvent(
                    event_id=f"evt_{stable_hash('evict', key, len(self.state.events))}",
                    timestamp_ms=self._clock_ms,
                    request_id=request.request_id,
                    event_type="evict",
                    cache_key=key,
                    content_hash=removed.content_hash,
                    semantic_type=removed.semantic_type,
                    token_count=removed.token_count,
                    policy_name=self.policy.name(),
                    reason=decision.reason,
                    estimated_latency_delta_ms=0.0,
                )
                request_events.append(event)
                self.state.append_event(event)
            cache_block = self._to_cache_block(block)
            self.state.add(cache_block)
            self.policy.on_admit(cache_block, self.state)
            admitted.append(cache_block.cache_key)
            event = self._event("admit", request, block, cache_block.cache_key, 0.0)
            request_events.append(event)
            self.state.append_event(event)
            if self.state.used_tokens > self.state.capacity_tokens:
                raise RuntimeError("cache capacity invariant violated")

        self.policy.update_from_feedback(request_events, self.state)
        uncached_tokens = sum(b.token_count for b in blocks) - tokens_saved
        estimated_ttft = self.base_ttft_ms + max(0, uncached_tokens) * self.token_latency_ms
        self.metrics.record_events(request_events)
        self.metrics.record_request(
            estimated_ttft,
            self.state.used_tokens,
            request_id=request.request_id,
            semantic_weights=self.policy.export_state(),
            events=request_events,
        )
        return RequestSimulationResult(
            request_id=request.request_id,
            policy_name=self.policy.name(),
            hits=hits,
            misses=misses,
            hit_keys=hit_keys,
            missed_hashes=missed_hashes,
            tokens_saved=tokens_saved,
            estimated_ttft_ms=round(estimated_ttft, 4),
            admitted_blocks=admitted,
            evicted_blocks=evicted,
            rejected_blocks=rejected,
            decision_id=decision.decision_id if decision else None,
        )

    def _to_cache_block(self, block: PromptBlock) -> CacheBlock:
        return CacheBlock(
            cache_key=f"ck_{block.content_hash}",
            content_hash=block.content_hash,
            semantic_type=block.semantic_type,
            token_count=block.token_count,
            memory_bytes_estimate=block.token_count * self.bytes_per_token,
            created_at_ms=self._clock_ms,
            last_accessed_at_ms=self._clock_ms,
            access_count=0,
            session_ids={block.session_id} if block.session_id else set(),
            tenant_id=block.tenant_id,
            estimated_latency_saved_ms=block.token_count * self.token_latency_ms,
            metadata=block.metadata,
        )

    def _event(
        self,
        event_type: str,
        request: PromptRequest,
        block: PromptBlock,
        cache_key: str | None,
        latency_delta: float,
        reason: str | None = None,
    ) -> CacheEvent:
        return CacheEvent(
            event_id=f"evt_{stable_hash(event_type, request.request_id, block.content_hash, len(self.state.events))}",
            timestamp_ms=self._clock_ms,
            request_id=request.request_id,
            event_type=event_type,  # type: ignore[arg-type]
            cache_key=cache_key,
            content_hash=block.content_hash,
            semantic_type=block.semantic_type,
            token_count=block.token_count,
            policy_name=self.policy.name(),
            reason=reason,
            estimated_latency_delta_ms=latency_delta,
        )

    def summary(self):
        return self.metrics.summary(self.policy.export_state())
