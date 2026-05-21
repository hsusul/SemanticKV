from __future__ import annotations

from typing import AsyncIterator, Callable

import httpx


async def iter_upstream_bytes(
    response: httpx.Response,
    on_first_chunk: Callable[[], None],
    on_complete: Callable[[], None],
) -> AsyncIterator[bytes]:
    first = True
    try:
        async for chunk in response.aiter_bytes():
            if first:
                on_first_chunk()
                first = False
            yield chunk
    finally:
        on_complete()
        await response.aclose()

