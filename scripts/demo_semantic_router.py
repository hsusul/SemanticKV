#!/usr/bin/env python
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_kv.routing.models import RoutingConfig
from semantic_kv.routing.router import SemanticKVRouter, replicas_from_urls


def main() -> None:
    output_dir = Path("outputs/routing")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policies = ["round_robin", "semantic_locality"]
    summaries = []
    for policy_name in policies:
        router = SemanticKVRouter(
            RoutingConfig(
                replicas=replicas_from_urls(["mock://replica-a", "mock://replica-b"]),
                policy_name=policy_name,
                max_tracked_tokens_per_replica=1200,
            )
        )
        decisions = []
        for idx, payload in enumerate(_traffic()):
            session_id = f"session-{idx}"
            blocks = router._segment(payload, f"demo-{idx}", session_id, "demo-tenant")
            decision = router.route_blocks(f"demo-{idx}", blocks, session_id=session_id)
            router.update_replica_state(decision.selected_replica_id, blocks, session_id=session_id)
            decisions.append(decision)
        avg_overlap = sum(d.estimated_prefix_overlap_tokens for d in decisions) / len(decisions)
        summaries.append(
            {
                "policy": policy_name,
                "requests": len(decisions),
                "avg_estimated_prefix_overlap_tokens": round(avg_overlap, 3),
                "fallback_count": sum(1 for d in decisions if d.fallback_used),
            }
        )

    metrics_path = output_dir / f"{stamp}_routing_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report_path = output_dir / f"{stamp}_routing_summary.md"
    report_path.write_text(_markdown(summaries), encoding="utf-8")
    print("SemanticKV routing demo complete")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    for row in summaries:
        print(f"{row['policy']}: avg estimated overlap tokens={row['avg_estimated_prefix_overlap_tokens']}")


def _traffic() -> list[dict]:
    docs = [
        "CONTEXT: Shared policy A. " + "Refunds approvals exceptions. " * 20,
        "CONTEXT: Shared policy B. " + "Security reviews approvals exceptions. " * 20,
    ]
    requests = []
    order = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 3
    for idx, doc_index in enumerate(order):
        doc = docs[doc_index]
        requests.append(
            {
                "model": "mock-router-model",
                "messages": [
                    {"role": "user", "content": f"{doc}\n\nUSER: question {idx} with unique details {idx * 13}"},
                ],
            }
        )
    return requests


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# SemanticKV Routing Demo",
        "",
        "This demo is an in-process cache-locality routing simulation. It does not control backend KV-cache eviction.",
        "",
        "| Policy | Requests | Avg Estimated Prefix Overlap Tokens | Fallback Count |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['requests']} | {row['avg_estimated_prefix_overlap_tokens']} | {row['fallback_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
