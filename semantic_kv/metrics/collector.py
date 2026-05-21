from __future__ import annotations

from collections import defaultdict
from statistics import mean

from semantic_kv.cache.models import CacheEvent, PolicyMetrics, SemanticType


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[idx]


class MetricsCollector:
    def __init__(self, policy_name: str, capacity_tokens: int) -> None:
        self.policy_name = policy_name
        self.capacity_tokens = capacity_tokens
        self.events: list[CacheEvent] = []
        self.ttft_values: list[float] = []
        self.memory_values: list[int] = []
        self.timeseries: list[dict] = []
        self.requests_total = 0
        self.regret_total = 0.0

    def record_request(
        self,
        estimated_ttft_ms: float,
        memory_tokens_used: int,
        request_id: str | None = None,
        semantic_weights: dict[str, float] | None = None,
        events: list[CacheEvent] | None = None,
    ) -> None:
        self.requests_total += 1
        self.ttft_values.append(estimated_ttft_ms)
        self.memory_values.append(memory_tokens_used)
        event_list = events or []
        self.timeseries.append(
            {
                "step": self.requests_total,
                "request_id": request_id,
                "estimated_ttft_ms": estimated_ttft_ms,
                "memory_tokens_used": memory_tokens_used,
                "memory_utilization": memory_tokens_used / self.capacity_tokens if self.capacity_tokens else 0.0,
                "hits": sum(1 for event in event_list if event.event_type == "hit"),
                "misses": sum(1 for event in event_list if event.event_type == "miss"),
                "evictions": sum(1 for event in event_list if event.event_type == "evict"),
                "regrets": sum(1 for event in event_list if event.event_type == "regret"),
                "semantic_weights": semantic_weights or {},
            }
        )

    def record_events(self, events: list[CacheEvent]) -> None:
        self.events.extend(events)
        self.regret_total += sum(max(0.0, e.estimated_latency_delta_ms) for e in events if e.event_type == "regret")

    def summary(self, final_semantic_weights: dict[str, float] | None = None) -> PolicyMetrics:
        hits = [e for e in self.events if e.event_type == "hit"]
        misses = [e for e in self.events if e.event_type == "miss"]
        evictions = [e for e in self.events if e.event_type == "evict"]
        tokens_saved = sum(e.token_count for e in hits)
        by_type_hits: dict[SemanticType, int] = defaultdict(int)
        by_type_accesses: dict[SemanticType, int] = defaultdict(int)
        evictions_by_type: dict[SemanticType, int] = defaultdict(int)
        for event in hits + misses:
            if event.semantic_type is not None:
                by_type_accesses[event.semantic_type] += 1
                if event.event_type == "hit":
                    by_type_hits[event.semantic_type] += 1
        for event in evictions:
            if event.semantic_type is not None:
                evictions_by_type[event.semantic_type] += 1
        accesses = len(hits) + len(misses)
        return PolicyMetrics(
            policy_name=self.policy_name,
            requests_total=self.requests_total,
            hits_total=len(hits),
            misses_total=len(misses),
            evictions_total=len(evictions),
            hit_rate=(len(hits) / accesses) if accesses else 0.0,
            tokens_saved_total=tokens_saved,
            estimated_ttft_ms_avg=mean(self.ttft_values) if self.ttft_values else 0.0,
            estimated_ttft_ms_p50=percentile(self.ttft_values, 0.50),
            estimated_ttft_ms_p95=percentile(self.ttft_values, 0.95),
            memory_tokens_used_avg=mean(self.memory_values) if self.memory_values else 0.0,
            memory_utilization_avg=(mean(self.memory_values) / self.capacity_tokens) if self.memory_values else 0.0,
            eviction_regret_total=self.regret_total,
            hit_rate_by_type={
                t: by_type_hits[t] / by_type_accesses[t] for t in by_type_accesses
            },
            evictions_by_type=dict(evictions_by_type),
            final_semantic_weights=final_semantic_weights or {},
        )
