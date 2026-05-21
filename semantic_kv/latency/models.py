from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from semantic_kv.traces.models import TraceRequest


class LatencyModel(Protocol):
    def fit(self, requests: list[TraceRequest]) -> "LatencyModel":
        ...

    def predict(
        self,
        uncached_tokens: int,
        total_prompt_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...


@dataclass
class LinearLatencyModel:
    base_ms: float = 80.0
    ms_per_uncached_token: float = 0.08
    long_context_threshold_tokens: int | None = None
    long_context_penalty_ms_per_token: float = 0.0

    def fit(self, requests: list[TraceRequest]) -> "LinearLatencyModel":
        return self

    def predict(
        self,
        uncached_tokens: int,
        total_prompt_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        estimate = self.base_ms + max(0, uncached_tokens) * self.ms_per_uncached_token
        if self.long_context_threshold_tokens is not None and total_prompt_tokens is not None:
            excess = max(0, total_prompt_tokens - self.long_context_threshold_tokens)
            estimate += excess * self.long_context_penalty_ms_per_token
        return estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "base_ms": self.base_ms,
            "ms_per_uncached_token": self.ms_per_uncached_token,
            "long_context_threshold_tokens": self.long_context_threshold_tokens,
            "long_context_penalty_ms_per_token": self.long_context_penalty_ms_per_token,
        }


@dataclass
class CalibratedLatencyModel(LinearLatencyModel):
    samples_used: int = 0

    def fit(self, requests: list[TraceRequest]) -> "CalibratedLatencyModel":
        samples: list[tuple[float, float]] = []
        for request in requests:
            if request.observed_ttft_ms is None:
                continue
            uncached = request.observed_prefill_tokens
            if uncached is None:
                uncached = max(0, request.total_prompt_tokens - (request.observed_prefix_hit_tokens or 0))
            samples.append((float(uncached), float(request.observed_ttft_ms)))

        if len(samples) < 2:
            self.samples_used = len(samples)
            return self

        n = len(samples)
        sum_x = sum(x for x, _ in samples)
        sum_y = sum(y for _, y in samples)
        sum_xx = sum(x * x for x, _ in samples)
        sum_xy = sum(x * y for x, y in samples)
        denominator = (n * sum_xx) - (sum_x * sum_x)
        if denominator == 0:
            self.base_ms = sum_y / n
            self.ms_per_uncached_token = 0.0
        else:
            self.ms_per_uncached_token = ((n * sum_xy) - (sum_x * sum_y)) / denominator
            self.base_ms = (sum_y - (self.ms_per_uncached_token * sum_x)) / n
        self.ms_per_uncached_token = max(0.0, self.ms_per_uncached_token)
        self.base_ms = max(0.0, self.base_ms)
        self.samples_used = n
        return self

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["kind"] = "calibrated_linear"
        data["samples_used"] = self.samples_used
        return data

