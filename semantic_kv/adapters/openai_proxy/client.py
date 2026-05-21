from __future__ import annotations

from typing import Any

import httpx


class OpenAICompatibleClient:
    def __init__(
        self,
        upstream_base_url: str,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def post_json(self, endpoint: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self.upstream_base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            return await client.post(endpoint, json=payload, headers=headers)

    def stream_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[httpx.AsyncClient, httpx.Request]:
        client = httpx.AsyncClient(
            base_url=self.upstream_base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        )
        request = client.build_request("POST", endpoint, json=payload, headers=headers)
        return client, request

