#!/usr/bin/env python
from __future__ import annotations

import os


def main() -> None:
    upstream = os.getenv("SEMANTICKV_UPSTREAM_BASE_URL", "http://localhost:8001")
    port = os.getenv("SEMANTICKV_PROXY_PORT", "8010")
    print("SemanticKV OpenAI-compatible telemetry proxy")
    print("This proxy records traces only; it does not control backend KV-cache eviction.")
    print(f"Upstream: {upstream}")
    print()
    print("Run:")
    print(f"SEMANTICKV_UPSTREAM_BASE_URL={upstream} uvicorn apps.proxy.main:app --host 0.0.0.0 --port {port}")


if __name__ == "__main__":
    main()

