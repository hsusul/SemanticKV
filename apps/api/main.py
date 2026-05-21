from __future__ import annotations

from fastapi import Depends, FastAPI

from apps.api.dependencies import get_simulator
from semantic_kv.cache.models import ExperimentConfig, PromptRequest
from semantic_kv.cache.simulator import PrefixCacheSimulator
from semantic_kv.experiments.runner import ExperimentRunner

app = FastAPI(title="SemanticKV API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "semantic-kv-api", "version": "0.1.0"}


@app.post("/v1/simulate/request")
def simulate_request(request: PromptRequest, simulator: PrefixCacheSimulator = Depends(get_simulator)):
    return simulator.simulate_request(request)


@app.post("/v1/experiments/run")
def run_experiment(config: ExperimentConfig):
    result = ExperimentRunner(config).run(write_outputs=True)
    return {
        "experiment_id": result.experiment_id,
        "best_policy_by_ttft": result.best_policy_by_ttft,
        "best_policy_by_hit_rate": result.best_policy_by_hit_rate,
        "output_dir": result.output_dir,
        "metrics_by_policy": result.metrics_by_policy,
    }


@app.get("/v1/cache/state")
def cache_state(simulator: PrefixCacheSimulator = Depends(get_simulator)):
    return simulator.state.snapshot()


@app.get("/v1/metrics/summary")
def metrics_summary(simulator: PrefixCacheSimulator = Depends(get_simulator)):
    return simulator.summary()

