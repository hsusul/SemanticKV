from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator

from semantic_kv.traces.models import TraceRequest


class TraceSpan:
    def __init__(self, collector: "TraceCollector", request_context: dict[str, Any]) -> None:
        self.collector = collector
        self.request_context = request_context
        self.observed_output_tokens: int | None = None
        self.observed_prefix_hit_tokens: int | None = None
        self.observed_prefill_tokens: int | None = None
        self.backend_metadata: dict[str, Any] = {}
        self.response_text: str | None = None

    def set_observed_output_tokens(self, value: int | None) -> None:
        self.observed_output_tokens = value

    def set_observed_prefix_hit_tokens(self, value: int | None) -> None:
        self.observed_prefix_hit_tokens = value

    def set_observed_prefill_tokens(self, value: int | None) -> None:
        self.observed_prefill_tokens = value

    def set_backend_metadata(self, metadata: dict[str, Any]) -> None:
        self.backend_metadata.update(metadata)

    def set_response_text(self, text: str | None) -> None:
        self.response_text = text


class TraceCollector(ABC):
    @abstractmethod
    def start_request(self, **kwargs: Any) -> TraceSpan:
        raise NotImplementedError

    @abstractmethod
    def finish_request(self, span: TraceSpan, **kwargs: Any) -> TraceRequest:
        raise NotImplementedError

    @abstractmethod
    def record_error(self, span: TraceSpan, error: BaseException) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @contextmanager
    def trace_request(self, **kwargs: Any) -> Iterator[TraceSpan]:
        span = self.start_request(**kwargs)
        try:
            yield span
        except BaseException as exc:
            self.record_error(span, exc)
            raise
        else:
            self.finish_request(span)

