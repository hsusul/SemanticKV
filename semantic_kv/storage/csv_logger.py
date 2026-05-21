from __future__ import annotations

import csv
from pathlib import Path

from semantic_kv.cache.models import PolicyMetrics


def write_metrics_csv(path: str | Path, metrics: dict[str, PolicyMetrics]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.model_dump(mode="json") for m in metrics.values()]
    fieldnames = [
        "policy_name",
        "requests_total",
        "hits_total",
        "misses_total",
        "evictions_total",
        "hit_rate",
        "tokens_saved_total",
        "estimated_ttft_ms_avg",
        "estimated_ttft_ms_p50",
        "estimated_ttft_ms_p95",
        "memory_utilization_avg",
        "eviction_regret_total",
        "relative_ttft_improvement_vs_lru",
        "relative_hit_rate_improvement_vs_lru",
        "hit_rate_by_type",
        "evictions_by_type",
        "final_semantic_weights",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
