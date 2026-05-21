from __future__ import annotations

from pathlib import Path

from semantic_kv.cache.models import ExperimentResult


def render_summary_markdown(result: ExperimentResult) -> str:
    lines = [
        f"# SemanticKV Experiment: {result.experiment_id}",
        "",
        "## Config",
        "",
        f"- Seed: `{result.config.seed}`",
        f"- Workloads: `{', '.join(result.config.workload_families)}`",
        f"- Cache capacity tokens: `{result.config.cache_capacity_tokens}`",
        f"- TTFT model: `base={result.config.base_ttft_ms}ms + uncached_tokens*{result.config.token_latency_ms}ms`",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Hit Rate | Tokens Saved | Avg Est. TTFT ms | P95 Est. TTFT ms | Evictions | Regret | Memory Util. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in sorted(result.metrics_by_policy.values(), key=lambda m: m.estimated_ttft_ms_avg):
        lines.append(
            f"| {metric.policy_name} | {metric.hit_rate:.3f} | {metric.tokens_saved_total} | "
            f"{metric.estimated_ttft_ms_avg:.2f} | {metric.estimated_ttft_ms_p95:.2f} | "
            f"{metric.evictions_total} | {metric.eviction_regret_total:.2f} | {metric.memory_utilization_avg:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Relative To LRU",
            "",
            "| Policy | Avg TTFT Improvement vs LRU | Hit Rate Improvement vs LRU |",
            "|---|---:|---:|",
        ]
    )
    for metric in sorted(result.metrics_by_policy.values(), key=lambda m: m.policy_name):
        ttft = metric.relative_ttft_improvement_vs_lru
        hit = metric.relative_hit_rate_improvement_vs_lru
        lines.append(
            f"| {metric.policy_name} | {ttft * 100:.2f}% | {hit * 100:.2f}% |"
            if ttft is not None and hit is not None
            else f"| {metric.policy_name} | n/a | n/a |"
        )
    lines.extend(
        [
            "## Best Policies",
            "",
            f"- Best by estimated TTFT: `{result.best_policy_by_ttft}`",
            f"- Best by hit rate: `{result.best_policy_by_hit_rate}`",
            "",
            "## Adaptive Semantic Weights",
            "",
            "## Generated Plots",
            "",
            "- `policy_comparison_avg_ttft.png`",
            "- `policy_comparison_hit_rate.png`",
            "- `tokens_saved_by_policy.png`",
            "- `evictions_by_semantic_type.png`",
            "- `adaptive_weights_over_time.png` when adaptive data exists",
            "- `memory_utilization_over_time.png`",
            "",
        ]
    )
    adaptive = result.metrics_by_policy.get("adaptive_semantic")
    if adaptive and adaptive.final_semantic_weights:
        for key, value in sorted(adaptive.final_semantic_weights.items()):
            lines.append(f"- `{key}`: {value:.4f}")
    else:
        lines.append("- Adaptive policy was not included.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "TTFT values are simulated estimates, not measured GPU serving latency. "
            "Use policy deltas and workload-specific trends as the main signal. "
            "Adaptive semantic eviction should help most when shared templates, system prompts, retrieved context, or tool schemas compete with one-off user blocks under memory pressure.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: str | Path, result: ExperimentResult) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_summary_markdown(result), encoding="utf-8")
