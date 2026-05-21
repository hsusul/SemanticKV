from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from semantic_kv.cache.models import ExperimentConfig, ExperimentResult, PromptRequest
from semantic_kv.cache.simulator import PrefixCacheSimulator
from semantic_kv.experiments.reports import render_summary_markdown, write_report
from semantic_kv.policies.base import build_policy
from semantic_kv.storage.csv_logger import write_metrics_csv
from semantic_kv.visualization.plots import write_experiment_plots
from semantic_kv.workloads.base import flatten_sessions
from semantic_kv.workloads.generators import build_workload_generator


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def build_requests(self) -> list[PromptRequest]:
        requests: list[PromptRequest] = []
        for idx, family in enumerate(self.config.workload_families):
            generator = build_workload_generator(family, self.config.seed + idx)
            sessions = generator.generate(self.config.num_sessions, self.config.requests_per_session)
            requests.extend(flatten_sessions(sessions))
        return sorted(requests, key=lambda r: (r.timestamp_ms, r.request_id))

    def run(self, write_outputs: bool = True) -> ExperimentResult:
        requests = self.build_requests()
        metrics = {}
        timeseries = {}
        for policy_name in self.config.policies:
            simulator = PrefixCacheSimulator(
                policy=build_policy(policy_name),
                capacity_tokens=self.config.cache_capacity_tokens,
                token_latency_ms=self.config.token_latency_ms,
                base_ttft_ms=self.config.base_ttft_ms,
            )
            for request in requests:
                simulator.simulate_request(request)
            metrics[policy_name] = simulator.summary()
            timeseries[policy_name] = simulator.metrics.timeseries

        self._add_relative_lru_metrics(metrics)
        best_ttft = min(metrics.values(), key=lambda m: m.estimated_ttft_ms_avg).policy_name
        best_hit = max(metrics.values(), key=lambda m: m.hit_rate).policy_name
        output_dir = self._output_dir()
        result = ExperimentResult(
            experiment_id=self.config.experiment_id,
            config=self.config,
            metrics_by_policy=metrics,
            best_policy_by_ttft=best_ttft,
            best_policy_by_hit_rate=best_hit,
            output_dir=str(output_dir),
            timeseries_by_policy=timeseries,
        )
        result.summary = render_summary_markdown(result)
        if write_outputs:
            self.write_outputs(result)
        return result

    def write_outputs(self, result: ExperimentResult) -> None:
        output_dir = Path(result.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        write_metrics_csv(output_dir / "metrics.csv", result.metrics_by_policy)
        write_report(output_dir / "summary.md", result)
        write_experiment_plots(result, output_dir)

    def _output_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return Path(self.config.output_dir) / f"{stamp}_{self.config.experiment_id}"

    def _add_relative_lru_metrics(self, metrics) -> None:
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


def summary_table(result: ExperimentResult) -> str:
    rows = ["policy                hit_rate  tokens_saved  avg_ttft  p95_ttft  evictions"]
    for metric in sorted(result.metrics_by_policy.values(), key=lambda m: m.estimated_ttft_ms_avg):
        rows.append(
            f"{metric.policy_name:<21} {metric.hit_rate:>7.3f}  {metric.tokens_saved_total:>12}  "
            f"{metric.estimated_ttft_ms_avg:>8.2f}  {metric.estimated_ttft_ms_p95:>8.2f}  {metric.evictions_total:>9}"
        )
    return "\n".join(rows)
