# SemanticKV

SemanticKV is a production-style LLM serving/cache infrastructure project that simulates semantic-aware prefix/KV-cache eviction under memory pressure. It segments prompts into semantic blocks, compares FIFO/LRU/LFU/size-aware/static/adaptive policies, records simulated serving metrics, and generates reproducible reports and plots. This is not a chatbot; it is an evaluation harness and FastAPI middleware MVP for studying cache behavior in LLM serving systems.

All TTFT values are simulated estimates, not measured GPU latency:

```text
estimated_ttft_ms = base_ttft_ms + uncached_prompt_tokens * token_latency_ms
```

## Architecture

```mermaid
flowchart LR
    A[Prompt Request] --> B[Segmenter]
    B --> C[Semantic Block Classifier]
    C --> D[Cache Simulator]
    D --> E[Eviction Policy]
    E --> F[Metrics Collector]
    F --> G[Reports + Plots]
```

Core components:

- `semantic_kv/cache`: prefix-cache simulator, cache state, Pydantic models
- `semantic_kv/policies`: FIFO, LRU, LFU, size-aware LRU, static semantic, adaptive semantic
- `semantic_kv/segmentation`: rule-based prompt segmentation and semantic classification
- `semantic_kv/workloads`: deterministic synthetic serving workloads
- `semantic_kv/experiments`: experiment runner, reports, aggregate summaries
- `apps/api`: FastAPI service for interactive simulation

## Why Semantic-Aware Eviction?

Prefix/KV caching avoids recomputing shared prompt prefixes during the prefill phase of LLM inference. Under constrained cache memory, eviction policy determines whether reusable blocks stay resident or get displaced by one-off prompt text.

Traditional policies treat prompt blocks uniformly:

- `LRU` works well when recent reuse predicts future reuse.
- `LFU` works well when exact frequent blocks dominate.
- `FIFO` is simple but blind to reuse value.
- Size-aware LRU accounts for block cost but not semantic role.

SemanticKV tests a more serving-aware thesis: system prompts, templates, retrieved context, tool outputs, code snippets, legal evidence, assistant history, and one-off user queries have different reuse patterns. A policy that combines semantic type, reuse feedback, recency, frequency, estimated prefill savings, and memory cost can make better eviction decisions in some workloads.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
python scripts/run_all_experiments.py
```

The aggregate run writes:

```text
outputs/experiments/aggregate_summary.csv
outputs/experiments/aggregate_summary.md
```

Each experiment writes:

```text
outputs/experiments/<timestamp>_<experiment_id>/
  results.json
  metrics.csv
  summary.md
  policy_comparison_avg_ttft.png
  policy_comparison_hit_rate.png
  tokens_saved_by_policy.png
  evictions_by_semantic_type.png
  adaptive_weights_over_time.png
  memory_utilization_over_time.png
```

## Example Results

Latest deterministic aggregate run:

| Workload | Main Result |
|---|---|
| `low_memory_rag_stress` | Adaptive semantic eviction improves simulated TTFT vs LRU by about `35.66%`. |
| `workload_shift_static_vs_adaptive` | Adaptive semantic eviction improves simulated TTFT vs LRU by about `19.86%`. |
| `legal_review_long_context` | LRU wins; recency is a strong proxy for reuse in this workload. |
| `tool_agent_session_reuse` | LRU wins; session-local temporal locality dominates semantic weighting. |

These are simulated results from deterministic synthetic workloads. They are useful for comparing policy behavior, not for claiming real GPU serving speedups.

## When Adaptive Wins / When LRU Wins

Adaptive semantic eviction wins when high-value semantic blocks are reused across requests while unique user text creates cache churn. The strongest examples are low-memory RAG and workload-shift experiments, where reusable retrieved context or templates compete with large one-off user blocks.

LRU wins when reuse is mostly session-local and recent access is already the best signal. Legal review and tool-agent sessions in the current suite show this clearly. This is intentional: credible systems evaluation should show where a new policy helps and where a simpler baseline is still better.

## Run One Experiment

```bash
python scripts/run_experiment.py --config configs/experiments/low_memory_rag_stress.yaml
```

Stress configs:

- `configs/experiments/low_memory_rag_stress.yaml`
- `configs/experiments/shared_system_prompt_high_reuse.yaml`
- `configs/experiments/tool_agent_session_reuse.yaml`
- `configs/experiments/legal_review_long_context.yaml`
- `configs/experiments/workload_shift_static_vs_adaptive.yaml`

## Trace Replay and Calibration

The synthetic experiments use a configurable estimated TTFT model. Trace replay adds a more credible offline validation step: load serving-style request telemetry, optionally fit the latency model from observed TTFT/prefill-token fields, and replay the same request sequence through multiple cache policies.

Example:

```bash
python scripts/replay_trace.py \
  --trace examples/traces/rag_serving_trace.jsonl \
  --cache-token-budget 50000 \
  --policies lru adaptive_semantic static_semantic
```

With calibration:

```bash
python scripts/replay_trace.py \
  --trace examples/traces/rag_serving_trace.jsonl \
  --calibrate-from-trace \
  --cache-token-budget 50000
```

Trace replay outputs are written under:

```text
outputs/traces/<timestamp>_<output_name>/
  replay_results.json
  replay_metrics.csv
  replay_summary.md
  replay_policy_avg_ttft.png
  replay_tokens_saved.png
  replay_hit_rate.png
  observed_vs_simulated_ttft.png
```

Valid claim: “On replay of the same trace, adaptive semantic eviction would have preserved more reusable tokens under SemanticKV’s calibrated latency model.”

Invalid claim: “Adaptive semantic eviction produced this speedup on a real GPU backend.” The current repo does not deploy or modify vLLM, SGLang, Ray Serve, or GPU KV-cache internals.

## Fast Health Check

```bash
python scripts/check_project.py
```

This verifies package imports, experiment config presence, policy imports, and FastAPI app import without running the full benchmark suite.

## Run the API

```bash
uvicorn apps.api.main:app --reload
```

```bash
curl http://127.0.0.1:8000/health
```

Example simulation request:

```bash
curl -X POST http://127.0.0.1:8000/v1/simulate/request \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_001",
    "session_id": "sess_001",
    "messages": [
      {"role": "system", "content": "You are a support assistant."},
      {"role": "user", "content": "CONTEXT: Document: refund policy section.\n\nUSER: summarize the policy."}
    ]
  }'
```

## Limitations

- TTFT is simulated/estimated only.
- The simulator stores cache metadata, not real GPU KV tensors.
- There is no vLLM, SGLang, or Ray Serve integration yet.
- The semantic classifier is rule-based and transparent, not a trained model.
- Workloads are deterministic synthetic traces, not production serving logs.

## Future Integration Path

A credible real-serving validation path would:

1. Capture prompt prefix/cache telemetry from vLLM, SGLang, or Ray Serve.
2. Calibrate token latency and memory-cost estimates against measured prefill behavior.
3. Replay real traces through SemanticKV and compare predicted policy deltas.
4. Add backend-specific adapters only where cache admission/eviction hooks are exposed.
