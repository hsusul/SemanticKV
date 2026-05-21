from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from semantic_kv.adapters.openai_proxy.client import OpenAICompatibleClient
from semantic_kv.adapters.openai_proxy.models import OpenAIProxySettings
from semantic_kv.adapters.openai_proxy.streaming import iter_upstream_bytes
from semantic_kv.collectors.jsonl_collector import JsonlTraceCollector
from semantic_kv.utils.time import now_ms


class OpenAICompatibleTelemetryProxy:
    def __init__(
        self,
        settings: OpenAIProxySettings,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or OpenAICompatibleClient(
            settings.upstream_base_url,
            timeout_seconds=settings.timeout_seconds,
        )
        self.latest_trace_path: str | None = None

    async def forward_chat_completions(self, request: Request) -> Response:
        payload = await request.json()
        return await self._forward(request, "/v1/chat/completions", payload, messages=payload.get("messages") or [])

    async def forward_completions(self, request: Request) -> Response:
        payload = await request.json()
        prompt = payload.get("prompt", "")
        if isinstance(prompt, list):
            prompt_text = "\n\n".join(str(p) for p in prompt)
        else:
            prompt_text = str(prompt)
        return await self._forward(request, "/v1/completions", payload, prompt=prompt_text)

    def latest_trace_summary(self) -> dict[str, Any]:
        if self.latest_trace_path is None:
            return {"trace_path": None, "exists": False, "requests": 0}
        path = Path(self.latest_trace_path)
        requests = 0
        if path.exists():
            requests = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return {"trace_path": self.latest_trace_path, "exists": path.exists(), "requests": requests}

    async def _forward(
        self,
        request: Request,
        endpoint: str,
        payload: dict[str, Any],
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
    ) -> Response:
        stream = bool(payload.get("stream", False))
        model_name = str(payload.get("model", "unknown-model"))
        collector = self._collector()
        headers = self._forward_headers(request)
        span = collector.start_request(
            request_id=str(payload.get("request_id") or f"proxy_req_{now_ms()}"),
            prompt=prompt,
            messages=messages or [],
            model_name=model_name,
            session_id=request.headers.get("X-Session-ID"),
            tenant_id=request.headers.get("X-Tenant-ID"),
            metadata={
                "adapter": "openai_proxy",
                "backend_url": self.settings.upstream_base_url,
                "endpoint": endpoint,
                "stream": stream,
            },
        )
        started = now_ms()
        first_chunk_seen_at: int | None = None

        if not stream:
            try:
                upstream = await self.client.post_json(endpoint, payload, headers=headers)
                total_ms = float(now_ms() - started)
                span.set_backend_metadata(
                    {
                        "status_code": upstream.status_code,
                        "observed_ttft_ms": total_ms,
                        "observed_total_latency_ms": total_ms,
                    }
                )
                collector.finish_request(span)
                content_type = upstream.headers.get("content-type", "application/json")
                return Response(content=upstream.content, status_code=upstream.status_code, media_type=content_type)
            except BaseException as exc:
                collector.record_error(span, exc)
                raise
            finally:
                collector.close()

        client, upstream_request = self.client.stream_request(endpoint, payload, headers=headers)
        upstream_response = await client.send(upstream_request, stream=True)

        def on_first_chunk() -> None:
            nonlocal first_chunk_seen_at
            if first_chunk_seen_at is None:
                first_chunk_seen_at = now_ms()

        def on_complete() -> None:
            total_ms = float(now_ms() - started)
            ttft_ms = float((first_chunk_seen_at or now_ms()) - started)
            span.set_backend_metadata(
                {
                    "status_code": upstream_response.status_code,
                    "observed_ttft_ms": ttft_ms,
                    "observed_total_latency_ms": total_ms,
                }
            )
            collector.finish_request(span)
            collector.close()

        media_type = upstream_response.headers.get("content-type", "text/event-stream")
        return StreamingResponse(
            iter_upstream_bytes(upstream_response, on_first_chunk=on_first_chunk, on_complete=on_complete),
            status_code=upstream_response.status_code,
            media_type=media_type,
        )

    def _collector(self) -> JsonlTraceCollector:
        output_dir = Path(self.settings.trace_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.latest_trace_path is None:
            self.latest_trace_path = str(output_dir / f"{now_ms()}_openai_proxy_trace.jsonl")
        return JsonlTraceCollector(
            self.latest_trace_path,
            backend_name="openai_compatible_proxy",
            drop_raw_text=self.settings.redact_text,
            append=True,
        )

    def _forward_headers(self, request: Request) -> dict[str, str]:
        excluded = {"host", "content-length"}
        return {key: value for key, value in request.headers.items() if key.lower() not in excluded}

