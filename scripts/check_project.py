#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import semantic_kv
    from apps.api.main import app
    from semantic_kv.policies.adaptive_semantic import AdaptiveSemanticPolicy
    from semantic_kv.policies.fifo import FIFOPolicy
    from semantic_kv.policies.lfu import LFUPolicy
    from semantic_kv.policies.lru import LRUPolicy
    from semantic_kv.policies.size_aware_lru import SizeAwareLRUPolicy
    from semantic_kv.policies.static_semantic import StaticSemanticPolicy

    config_dir = Path("configs/experiments")
    if not config_dir.exists():
        raise SystemExit("configs/experiments does not exist")
    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit("configs/experiments contains no YAML configs")

    policies = [
        FIFOPolicy(),
        LRUPolicy(),
        LFUPolicy(),
        SizeAwareLRUPolicy(),
        StaticSemanticPolicy(),
        AdaptiveSemanticPolicy(),
    ]
    policy_names = ", ".join(policy.name() for policy in policies)
    routes = {route.path for route in app.routes}
    if "/health" not in routes:
        raise SystemExit("FastAPI app missing /health route")

    print("SemanticKV project check passed")
    print(f"Package version: {semantic_kv.__version__}")
    print(f"Experiment configs: {len(configs)}")
    print(f"Policies: {policy_names}")
    print("FastAPI app import: ok")


if __name__ == "__main__":
    main()
