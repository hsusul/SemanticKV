from pathlib import Path

from semantic_kv.cache.models import ExperimentConfig
from semantic_kv.experiments.runner import ExperimentRunner


def test_experiment_runner_writes_outputs(tmp_path: Path):
    config = ExperimentConfig(
        experiment_id="test",
        seed=3,
        policies=["lru", "adaptive_semantic"],
        workload_families=["repeated_system", "rag_shared_corpus"],
        num_sessions=4,
        requests_per_session=3,
        cache_capacity_tokens=120,
        output_dir=str(tmp_path),
    )
    result = ExperimentRunner(config).run(write_outputs=True)
    assert set(result.metrics_by_policy) == {"lru", "adaptive_semantic"}
    assert Path(result.output_dir, "results.json").exists()
    assert Path(result.output_dir, "metrics.csv").exists()
    assert Path(result.output_dir, "summary.md").exists()

