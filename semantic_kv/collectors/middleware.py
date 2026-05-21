from __future__ import annotations

from typing import Any, Callable

from semantic_kv.collectors.base import TraceCollector


def collect_generation_trace(
    collector: TraceCollector,
    backend_generate: Callable[[str], Any],
    prompt: str,
    model_name: str,
    session_id: str | None = None,
    tenant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    with collector.trace_request(
        prompt=prompt,
        model_name=model_name,
        session_id=session_id,
        tenant_id=tenant_id,
        metadata=metadata or {},
    ) as span:
        response = backend_generate(prompt)
        if isinstance(response, dict):
            span.set_response_text(response.get("text"))
            span.set_observed_output_tokens(response.get("output_tokens"))
            span.set_observed_prefix_hit_tokens(response.get("observed_prefix_hit_tokens"))
            span.set_observed_prefill_tokens(response.get("observed_prefill_tokens"))
            span.set_backend_metadata(response.get("metadata", {}))
        return response

