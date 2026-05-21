from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "semantic_kv_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_kv.cache.models import ExperimentResult


def write_experiment_plots(result: ExperimentResult, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        _bar_metric(result, output, "estimated_ttft_ms_avg", "Avg Estimated TTFT (ms)", "policy_comparison_avg_ttft.png"),
        _bar_metric(result, output, "hit_rate", "Cache Hit Rate", "policy_comparison_hit_rate.png"),
        _bar_metric(result, output, "tokens_saved_total", "Prefill Tokens Saved", "tokens_saved_by_policy.png"),
        _evictions_by_semantic_type(result, output),
        _memory_utilization(result, output),
    ]
    adaptive_path = _adaptive_weights(result, output)
    if adaptive_path:
        paths.append(adaptive_path)
    return [path for path in paths if path is not None]


def _policy_order(result: ExperimentResult) -> list[str]:
    return sorted(result.metrics_by_policy, key=lambda p: result.metrics_by_policy[p].estimated_ttft_ms_avg)


def _bar_metric(result: ExperimentResult, output: Path, field: str, ylabel: str, filename: str) -> Path:
    policies = _policy_order(result)
    values = [getattr(result.metrics_by_policy[p], field) for p in policies]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(policies, values, color="#476A6F")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by Policy")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _evictions_by_semantic_type(result: ExperimentResult, output: Path) -> Path:
    semantic_types = sorted(
        {
            str(t)
            for metric in result.metrics_by_policy.values()
            for t in metric.evictions_by_type
        }
    )
    policies = _policy_order(result)
    if not semantic_types:
        semantic_types = ["none"]
    bottoms = [0] * len(policies)
    fig, ax = plt.subplots(figsize=(10, 5))
    for semantic_type in semantic_types:
        values = []
        for policy in policies:
            evictions = result.metrics_by_policy[policy].evictions_by_type
            values.append(evictions.get(semantic_type, evictions.get(str(semantic_type), 0)))
        ax.bar(policies, values, bottom=bottoms, label=semantic_type)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("Evictions")
    ax.set_title("Evictions by Semantic Type")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output / "evictions_by_semantic_type.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _adaptive_weights(result: ExperimentResult, output: Path) -> Path | None:
    series = result.timeseries_by_policy.get("adaptive_semantic", [])
    rows = [(row["step"], row.get("semantic_weights", {})) for row in series if row.get("semantic_weights")]
    if not rows:
        return None
    by_type: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for step, weights in rows:
        for semantic_type, weight in weights.items():
            by_type[semantic_type].append((step, weight))
    fig, ax = plt.subplots(figsize=(10, 5))
    for semantic_type, points in sorted(by_type.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, label=semantic_type)
    ax.set_xlabel("Request step")
    ax.set_ylabel("Semantic weight")
    ax.set_title("Adaptive Semantic Weights Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    fig.tight_layout()
    path = output / "adaptive_weights_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _memory_utilization(result: ExperimentResult, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy, series in sorted(result.timeseries_by_policy.items()):
        if not series:
            continue
        ax.plot([row["step"] for row in series], [row["memory_utilization"] for row in series], label=policy)
    ax.set_xlabel("Request step")
    ax.set_ylabel("Memory utilization")
    ax.set_title("Cache Memory Utilization Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize="small", ncol=2)
    fig.tight_layout()
    path = output / "memory_utilization_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_trace_replay_plots(result: dict, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics_by_policy"]
    paths = [
        _trace_bar(metrics, output, "estimated_ttft_ms_avg", "Avg Estimated TTFT (ms)", "replay_policy_avg_ttft.png"),
        _trace_bar(metrics, output, "tokens_saved_total", "Tokens Saved", "replay_tokens_saved.png"),
        _trace_bar(metrics, output, "hit_rate", "Hit Rate", "replay_hit_rate.png"),
    ]
    observed = result.get("observed_summary") or {}
    if observed.get("observed_avg_ttft_ms") is not None:
        paths.append(_observed_vs_simulated_ttft(result, output))
    return paths


def _trace_policy_order(metrics: dict) -> list[str]:
    return sorted(metrics, key=lambda p: metrics[p]["estimated_ttft_ms_avg"])


def _trace_bar(metrics: dict, output: Path, field: str, ylabel: str, filename: str) -> Path:
    policies = _trace_policy_order(metrics)
    values = [metrics[p][field] for p in policies]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(policies, values, color="#4F6F52")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Trace Replay {ylabel} by Policy")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _observed_vs_simulated_ttft(result: dict, output: Path) -> Path:
    observed = result["observed_summary"]["observed_avg_ttft_ms"]
    lru = result["metrics_by_policy"].get("lru", {}).get("estimated_ttft_ms_avg")
    labels = ["observed", "simulated_lru"]
    values = [observed, lru or 0.0]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, values, color=["#8B5E34", "#476A6F"])
    ax.set_ylabel("Avg TTFT ms")
    ax.set_title("Observed vs Simulated LRU TTFT")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output / "observed_vs_simulated_ttft.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
