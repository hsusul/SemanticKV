from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from semantic_kv.cache.models import PolicyMetrics, PromptBlock, PromptRequest
from semantic_kv.cache.simulator import PrefixCacheSimulator
from semantic_kv.latency.models import CalibratedLatencyModel, LatencyModel, LinearLatencyModel
from semantic_kv.policies.base import build_policy
from semantic_kv.traces.io import load_trace
from semantic_kv.traces.models import ServingTrace, TraceReplayConfig, TraceRequest
from semantic_kv.utils.hashing import normalize_text, stable_hash
from semantic_kv.visualization.plots import write_trace_replay_plots


class TraceReplayRunner:
    def __init__(self, config: TraceReplayConfig) -> None:
        self.config = config

    def run(self, write_outputs: bool = True) -> dict[str, Any]:
        trace = load_trace(self.config.trace_path)
        latency_model = self._build_latency_model(trace)
        metrics_by_policy: dict[str, PolicyMetrics] = {}
        timeseries_by_policy: dict[str, list[dict[str, Any]]] = {}
        observed_summary = self._observed_summary(trace)

        for policy_name in self.config.policies:
            simulator = PrefixCacheSimulator(
                policy=build_policy(policy_name),
                capacity_tokens=self.config.cache_token_budget,
                token_latency_ms=latency_model.to_dict()["ms_per_uncached_token"],
                base_ttft_ms=latency_model.to_dict()["base_ms"],
            )
            for request in sorted(trace.requests, key=lambda r: (r.timestamp_ms, r.request_id)):
                simulator.process_blocks(self._to_prompt_request(request), self._to_prompt_blocks(request))
            metrics_by_policy[policy_name] = simulator.summary()
            timeseries_by_policy[policy_name] = simulator.metrics.timeseries

        self._add_relative_lru_metrics(metrics_by_policy)
        output_dir = self._output_dir()
        result = {
            "trace_id": trace.trace_id,
            "backend_name": trace.backend_name,
            "model_name": trace.model_name,
            "request_count": len(trace.requests),
            "cache_token_budget": self.config.cache_token_budget,
            "latency_model": latency_model.to_dict(),
            "policies": self.config.policies,
            "metrics_by_policy": {k: v.model_dump(mode="json") for k, v in metrics_by_policy.items()},
            "observed_summary": observed_summary,
            "observed_vs_simulated": self._observed_vs_simulated(observed_summary, metrics_by_policy),
            "timeseries_by_policy": timeseries_by_policy,
            "output_dir": str(output_dir),
        }
        if write_outputs:
            self.write_outputs(result, metrics_by_policy, output_dir)
        return result

    def write_outputs(self, result: dict[str, Any], metrics_by_policy: dict[str, PolicyMetrics], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "replay_results.json").write_text(_json_dumps(result), encoding="utf-8")
        self._write_metrics_csv(output_dir / "replay_metrics.csv", metrics_by_policy)
        (output_dir / "replay_summary.md").write_text(render_replay_summary(result), encoding="utf-8")
        write_trace_replay_plots(result, output_dir)

    def _build_latency_model(self, trace: ServingTrace) -> LatencyModel:
        cfg = self.config.latency_model
        if self.config.calibrate_from_trace:
            return CalibratedLatencyModel(
                base_ms=cfg.base_ms,
                ms_per_uncached_token=cfg.ms_per_uncached_token,
                long_context_threshold_tokens=cfg.long_context_threshold_tokens,
                long_context_penalty_ms_per_token=cfg.long_context_penalty_ms_per_token,
            ).fit(trace.requests)
        return LinearLatencyModel(
            base_ms=cfg.base_ms,
            ms_per_uncached_token=cfg.ms_per_uncached_token,
            long_context_threshold_tokens=cfg.long_context_threshold_tokens,
            long_context_penalty_ms_per_token=cfg.long_context_penalty_ms_per_token,
        )

    def _to_prompt_request(self, request: TraceRequest) -> PromptRequest:
        return PromptRequest(
            request_id=request.request_id,
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            timestamp_ms=request.timestamp_ms,
            model_name=request.model_name,
            metadata={"trace_replay": True, **request.metadata},
        )

    def _to_prompt_blocks(self, request: TraceRequest) -> list[PromptBlock]:
        blocks: list[PromptBlock] = []
        for position, block in enumerate(request.blocks):
            text = block.text or block.content_hash or block.block_id or f"{block.semantic_type.value}:{position}"
            content_hash = block.content_hash or stable_hash(block.semantic_type.value, normalize_text(text), block.token_count)
            blocks.append(
                PromptBlock(
                    block_id=block.block_id or f"trace_blk_{stable_hash(request.request_id, position, content_hash)}",
                    request_id=request.request_id,
                    session_id=request.session_id,
                    tenant_id=request.tenant_id,
                    position=block.position if block.position is not None else position,
                    text=text,
                    token_count=block.token_count,
                    semantic_type=block.semantic_type,
                    source="trace",
                    content_hash=content_hash,
                    metadata=block.metadata,
                )
            )
        return blocks

    def _observed_summary(self, trace: ServingTrace) -> dict[str, float | int | None]:
        ttft_values = [r.observed_ttft_ms for r in trace.requests if r.observed_ttft_ms is not None]
        prefill = [r.observed_prefill_tokens for r in trace.requests if r.observed_prefill_tokens is not None]
        hit_tokens = [r.observed_prefix_hit_tokens for r in trace.requests if r.observed_prefix_hit_tokens is not None]
        return {
            "requests_with_observed_ttft": len(ttft_values),
            "observed_avg_ttft_ms": mean(ttft_values) if ttft_values else None,
            "observed_prefill_tokens_total": sum(prefill) if prefill else None,
            "observed_prefix_hit_tokens_total": sum(hit_tokens) if hit_tokens else None,
        }

    def _observed_vs_simulated(
        self,
        observed: dict[str, float | int | None],
        metrics_by_policy: dict[str, PolicyMetrics],
    ) -> dict[str, Any]:
        lru = metrics_by_policy.get("lru")
        if lru is None:
            return {}
        return {
            "observed_avg_ttft_ms": observed.get("observed_avg_ttft_ms"),
            "lru_simulated_avg_ttft_ms": lru.estimated_ttft_ms_avg,
            "observed_prefix_hit_tokens_total": observed.get("observed_prefix_hit_tokens_total"),
            "lru_simulated_tokens_saved_total": lru.tokens_saved_total,
        }

    def _add_relative_lru_metrics(self, metrics: dict[str, PolicyMetrics]) -> None:
        lru = metrics.get("lru")
        if lru is None:
            return
        for metric in metrics.values():
            if lru.estimated_ttft_ms_avg:
                metric.relative_ttft_improvement_vs_lru = (
                    (lru.estimated_ttft_ms_avg - metric.estimated_ttft_ms_avg) / lru.estimated_ttft_ms_avg
                )
            if lru.hit_rate:
                metric.relative_hit_rate_improvement_vs_lru = (metric.hit_rate - lru.hit_rate) / lru.hit_rate

    def _write_metrics_csv(self, path: Path, metrics: dict[str, PolicyMetrics]) -> None:
        fieldnames = [
            "policy_name",
            "hit_rate",
            "tokens_saved_total",
            "estimated_ttft_ms_avg",
            "estimated_ttft_ms_p50",
            "estimated_ttft_ms_p95",
            "evictions_total",
            "eviction_regret_total",
            "relative_ttft_improvement_vs_lru",
            "relative_hit_rate_improvement_vs_lru",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics.values():
                row = metric.model_dump(mode="json")
                writer.writerow({key: row.get(key) for key in fieldnames})

    def _output_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return Path(self.config.output_dir) / f"{stamp}_{self.config.output_name}"


def render_replay_summary(result: dict[str, Any]) -> str:
    metrics = result["metrics_by_policy"]
    lines = [
        f"# Trace Replay: {result['trace_id']}",
        "",
        "All TTFT values are simulated or calibrated estimates from offline trace replay, not measured deployment results.",
        "",
        "## Trace",
        "",
        f"- Backend label: `{result['backend_name']}`",
        f"- Model: `{result.get('model_name')}`",
        f"- Requests: `{result['request_count']}`",
        f"- Cache token budget: `{result['cache_token_budget']}`",
        f"- Latency model: `{result['latency_model']}`",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Hit Rate | Tokens Saved | Avg Estimated TTFT ms | P95 Estimated TTFT ms | Evictions | TTFT Improvement vs LRU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in sorted(metrics.items(), key=lambda item: item[1]["estimated_ttft_ms_avg"]):
        improvement = metric.get("relative_ttft_improvement_vs_lru")
        improvement_text = "n/a" if improvement is None else f"{improvement * 100:.2f}%"
        lines.append(
            f"| {name} | {metric['hit_rate']:.3f} | {metric['tokens_saved_total']} | "
            f"{metric['estimated_ttft_ms_avg']:.2f} | {metric['estimated_ttft_ms_p95']:.2f} | "
            f"{metric['evictions_total']} | {improvement_text} |"
        )
    observed = result.get("observed_summary") or {}
    if observed.get("observed_avg_ttft_ms") is not None:
        lines.extend(
            [
                "",
                "## Observed vs Simulated LRU-Like Replay",
                "",
                f"- Observed avg TTFT: `{observed['observed_avg_ttft_ms']:.2f} ms`",
                f"- Simulated LRU avg TTFT: `{result['observed_vs_simulated'].get('lru_simulated_avg_ttft_ms'):.2f} ms`",
                f"- Observed prefix-hit tokens: `{observed.get('observed_prefix_hit_tokens_total')}`",
                f"- Simulated LRU tokens saved: `{result['observed_vs_simulated'].get('lru_simulated_tokens_saved_total')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These results answer an offline counterfactual: on replay of the same trace, a policy would have preserved the reported number of reusable tokens under SemanticKV's simulator and latency model. They do not mean the adaptive policy was deployed on the source backend.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_dumps(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)

