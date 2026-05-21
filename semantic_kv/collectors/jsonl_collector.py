from __future__ import annotations

from pathlib import Path
from typing import Any

from semantic_kv.collectors.base import TraceCollector, TraceSpan
from semantic_kv.segmentation.segmenter import PromptSegmenter
from semantic_kv.traces.anonymize import anonymize_block
from semantic_kv.traces.models import TracePromptBlock, TraceRequest
from semantic_kv.utils.hashing import stable_hash
from semantic_kv.utils.time import now_ms


class JsonlTraceCollector(TraceCollector):
    def __init__(
        self,
        output_path: str | Path,
        backend_name: str = "unknown_backend",
        model_name: str | None = None,
        drop_raw_text: bool = True,
        append: bool = False,
        segmenter: PromptSegmenter | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend_name = backend_name
        self.model_name = model_name
        self.drop_raw_text = drop_raw_text
        self.segmenter = segmenter or PromptSegmenter()
        self._handle = self.output_path.open("a" if append else "w", encoding="utf-8")
        self._request_counter = 0
        self.latest_request: TraceRequest | None = None
        self.errors: list[dict[str, Any]] = []

    def start_request(self, **kwargs: Any) -> TraceSpan:
        started_at_ms = now_ms()
        request_id = kwargs.get("request_id") or f"trace_req_{stable_hash(started_at_ms, self._request_counter)}"
        self._request_counter += 1
        return TraceSpan(
            self,
            {
                "request_id": request_id,
                "started_at_ms": started_at_ms,
                "prompt": kwargs.get("prompt"),
                "messages": kwargs.get("messages"),
                "model_name": kwargs.get("model_name") or self.model_name,
                "session_id": kwargs.get("session_id"),
                "tenant_id": kwargs.get("tenant_id"),
                "metadata": kwargs.get("metadata") or {},
            },
        )

    def finish_request(self, span: TraceSpan, **kwargs: Any) -> TraceRequest:
        finished_at_ms = now_ms()
        metadata = {
            "backend_name": self.backend_name,
            **span.request_context.get("metadata", {}),
            **span.backend_metadata,
        }
        if span.observed_output_tokens is not None:
            metadata["observed_output_tokens"] = span.observed_output_tokens
        trace_request = TraceRequest(
            request_id=span.request_context["request_id"],
            timestamp_ms=span.request_context["started_at_ms"],
            session_id=span.request_context.get("session_id"),
            tenant_id=span.request_context.get("tenant_id"),
            model_name=span.request_context.get("model_name"),
            blocks=self._segment_to_trace_blocks(span),
            observed_ttft_ms=_coalesce(kwargs.get("observed_ttft_ms"), span.backend_metadata.get("observed_ttft_ms")),
            observed_total_latency_ms=_coalesce(
                kwargs.get("observed_total_latency_ms"),
                span.backend_metadata.get("observed_total_latency_ms"),
                float(finished_at_ms - span.request_context["started_at_ms"]),
            ),
            observed_prefix_hit_tokens=_coalesce(
                kwargs.get("observed_prefix_hit_tokens"),
                span.observed_prefix_hit_tokens,
                span.backend_metadata.get("observed_prefix_hit_tokens"),
            ),
            observed_prefill_tokens=_coalesce(
                kwargs.get("observed_prefill_tokens"),
                span.observed_prefill_tokens,
                span.backend_metadata.get("observed_prefill_tokens"),
            ),
            metadata=metadata,
        )
        self._handle.write(trace_request.model_dump_json())
        self._handle.write("\n")
        self._handle.flush()
        self.latest_request = trace_request
        return trace_request

    def record_error(self, span: TraceSpan, error: BaseException) -> None:
        self.errors.append(
            {
                "request_id": span.request_context.get("request_id"),
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp_ms": now_ms(),
            }
        )

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def _segment_to_trace_blocks(self, span: TraceSpan) -> list[TracePromptBlock]:
        from semantic_kv.cache.models import PromptRequest

        request = PromptRequest(
            request_id=span.request_context["request_id"],
            session_id=span.request_context.get("session_id"),
            tenant_id=span.request_context.get("tenant_id"),
            timestamp_ms=span.request_context["started_at_ms"],
            raw_prompt=span.request_context.get("prompt"),
            messages=span.request_context.get("messages") or [],
            model_name=span.request_context.get("model_name"),
            metadata=span.request_context.get("metadata") or {},
        )
        trace_blocks: list[TracePromptBlock] = []
        for block in self.segmenter.segment(request):
            trace_block = TracePromptBlock(
                block_id=block.block_id,
                content_hash=block.content_hash,
                semantic_type=block.semantic_type,
                token_count=block.token_count,
                text=block.text,
                position=block.position,
                metadata=block.metadata,
            )
            if self.drop_raw_text:
                trace_block = anonymize_block(trace_block, drop_text=True)
            trace_blocks.append(trace_block)
        return trace_blocks


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None

