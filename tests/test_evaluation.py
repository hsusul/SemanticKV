from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from semantic_kv.cache.models import ExperimentConfig
from semantic_kv.cache.simulator import PrefixCacheSimulator
from semantic_kv.experiments.runner import ExperimentRunner
from semantic_kv.policies.adaptive_semantic import AdaptiveSemanticPolicy
from semantic_kv.visualization.plots import write_experiment_plots
from semantic_kv.workloads.base import flatten_sessions
from semantic_kv.workloads.generators import build_workload_generator


STRESS_WORKLOADS = [
    "low_memory_rag_stress",
    "shared_system_prompt_high_reuse",
    "tool_agent_session_reuse",
    "legal_review_long_context",
    "workload_shift_static_vs_adaptive",
]


def test_new_stress_workloads_are_deterministic():
    for name in STRESS_WORKLOADS:
        a = build_workload_generator(name, 17).generate(4, 3)
        b = build_workload_generator(name, 17).generate(4, 3)
        assert [r.model_dump() for r in flatten_sessions(a)] == [r.model_dump() for r in flatten_sessions(b)]


def test_low_memory_stress_creates_evictions():
    config = ExperimentConfig(
        experiment_id="low_memory_test",
        seed=9,
        policies=["lru"],
        workload_families=["low_memory_rag_stress"],
        num_sessions=8,
        requests_per_session=4,
        cache_capacity_tokens=2700,
    )
    result = ExperimentRunner(config).run(write_outputs=False)
    assert result.metrics_by_policy["lru"].evictions_total > 0


def test_plot_generation_writes_png_files(tmp_path: Path):
    config = ExperimentConfig(
        experiment_id="plot_test",
        seed=4,
        policies=["lru", "adaptive_semantic"],
        workload_families=["shared_system_prompt_high_reuse"],
        num_sessions=5,
        requests_per_session=3,
        cache_capacity_tokens=150,
        output_dir=str(tmp_path),
    )
    result = ExperimentRunner(config).run(write_outputs=False)
    paths = write_experiment_plots(result, tmp_path)
    assert paths
    assert all(path.exists() and path.suffix == ".png" for path in paths)
    assert (tmp_path / "adaptive_weights_over_time.png").exists()


def test_adaptive_policy_records_semantic_weight_history():
    generator = build_workload_generator("workload_shift_static_vs_adaptive", 3)
    requests = flatten_sessions(generator.generate(8, 3))
    simulator = PrefixCacheSimulator(AdaptiveSemanticPolicy(), capacity_tokens=2700)
    for request in requests:
        simulator.simulate_request(request)
    weights = [row["semantic_weights"] for row in simulator.metrics.timeseries]
    assert weights
    assert any(row.get("retrieved_context") != weights[0].get("retrieved_context") for row in weights[1:])


def test_aggregate_experiment_script_outputs_files(tmp_path: Path):
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "outputs"
    config_dir.mkdir()
    config = {
        "experiment_id": "aggregate_smoke",
        "seed": 5,
        "policies": ["lru", "adaptive_semantic"],
        "workload_families": ["shared_system_prompt_high_reuse"],
        "num_sessions": 4,
        "requests_per_session": 3,
        "cache_capacity_tokens": 150,
        "output_dir": str(output_dir),
    }
    (config_dir / "aggregate_smoke.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert (output_dir / "aggregate_summary.csv").exists()
    assert (output_dir / "aggregate_summary.md").exists()
