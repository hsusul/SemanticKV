# SemanticKV PRD

## 1. Executive Summary

SemanticKV is a production-style adaptive prefix-cache simulator and serving middleware for LLM workloads. It models prompt and KV-cache blocks as semantically typed units, learns eviction priorities online from cache feedback, and compares adaptive semantic-aware eviction against baseline cache policies such as LRU, LFU, FIFO, size-aware LRU, and static semantic weighting.

Modern LLM serving systems rely on prefix and KV caching to avoid recomputing repeated prompt prefixes. This matters because prefill work is expensive, long-context applications are increasingly common, and GPU memory pressure forces serving systems to decide which cached blocks deserve limited cache capacity. Treating every cached block as equivalent wastes memory on low-reuse prompt regions while evicting blocks that may save substantial prefill latency later.

SemanticKV is not a generic LLM application or chatbot. It is an AI infrastructure project focused on cache policy simulation, request middleware, observability, and reproducible evaluation. Its primary product surface is a simulator, policy engine, experiment runner, FastAPI service, metrics layer, and research-style reports.

The core technical thesis is: prompt blocks with different semantic roles have measurably different reuse value, and an adaptive eviction policy that combines semantic type, recency, frequency, estimated prefill savings, memory cost, and online regret feedback can improve estimated TTFT, cache hit rate, and tokens saved under memory pressure.

## 2. Problem Statement

LLM inference consists of prefill and decode phases. During prefill, the model processes the input prompt and produces key-value tensors for attention layers. Prefix/KV caching stores previously computed KV states for prompt prefixes so future requests sharing the same prefix can skip recomputation for matching tokens.

Prefix caching is valuable for workloads with repeated templates, system prompts, conversation history, retrieved documents, tool outputs, or organization-specific boilerplate. However, KV cache memory is large and normally lives in constrained GPU or high-performance memory. As concurrent sessions, context length, and document-heavy workflows grow, the cache cannot retain every block indefinitely. Eviction policy becomes a first-order serving decision.

Traditional cache policies are insufficient:

- `LRU` assumes recently used blocks are most valuable, but repeated system prompts or shared RAG chunks may become valuable after gaps.
- `LFU` favors globally frequent blocks, but can retain stale blocks after workload shifts.
- `FIFO` is simple and predictable, but ignores reuse value.
- Size-blind policies may retain large low-value blocks while evicting small high-value blocks.
- Static semantic rules can encode domain intuition but cannot adapt when the workload changes.

Prompt segments have different reuse behavior. System prompts and templates are often shared across many requests. User queries may be mostly unique. Retrieved context may be reused across sessions when many users ask about the same corpus. Assistant history may be reused within a session but rarely across sessions. Tool outputs may be valuable if deterministic and repeated, or useless if request-specific. A cache policy that understands these semantic types can make better tradeoffs under memory pressure.

## 3. Goals

### Product Goals

- Provide a credible AI infrastructure project that demonstrates LLM serving knowledge without requiring custom CUDA or vLLM internals in the MVP.
- Let users run reproducible experiments comparing eviction policies across realistic workload families.
- Expose a FastAPI service that can sit in front of a future LLM backend as request middleware.
- Produce readable reports and dashboards that explain why one policy outperforms another.
- Make the system easy to run locally with Docker Compose and easy to extend with new policies, workloads, and backends.

### Technical Goals

- Simulate LLM prefix-cache behavior under realistic multi-session workloads.
- Segment prompts into semantically meaningful blocks.
- Classify prompt blocks into semantic types such as `system`, `template`, `user_query`, `retrieved_context`, `tool_output`, `assistant_history`, `code_context`, `legal_evidence`, and `other`.
- Implement multiple eviction policies behind a shared interface.
- Learn adaptive semantic weights online from hits, misses, evictions, and regret signals.
- Estimate TTFT, prefix reuse depth, prefill tokens saved, and memory pressure effects.
- Export metrics suitable for Prometheus and local dashboards.
- Persist experiment outputs as JSON, CSV, or Parquet for reproducible analysis.

## 4. Non-Goals

- Do not build a generic chatbot.
- Do not require modifying CUDA kernels, vLLM internals, SGLang internals, or Ray Serve internals in the MVP.
- Do not claim exact GPU KV-cache performance in the MVP. The simulator estimates latency and memory pressure using configurable cost models.
- Do not train a foundation model.
- Do not implement unsafe hidden chain-of-thought handling. Reasoning traces may appear only as synthetic/private block labels in test workloads and must not be exposed as real user hidden reasoning.
- Do not optimize for production multi-node serving in the first version.
- Do not make dashboards the center of the project at the expense of algorithmic evaluation.

## 5. User Personas

### AI Infrastructure Engineer

Evaluates cache policies for LLM serving platforms. Wants clean policy abstractions, realistic workloads, memory-pressure simulation, and metrics that map to serving concerns such as TTFT, prefill cost, and cache utilization.

### MLE Experimenting With Serving Optimization

Tests whether semantic-aware eviction improves performance for application-specific LLM workloads. Needs configurable workloads, reproducible experiment runs, and interpretable policy behavior.

### Platform Engineer Monitoring Serving Latency

Operates LLM request gateways and wants visibility into cache hit rates, estimated latency impact, memory pressure, and degradation under workload spikes.

### Research Engineer Comparing Adaptive Policies

Runs controlled experiments across policy variants. Needs baselines, ablations, static semantic weighting, adaptive feedback loops, workload mismatch tests, and repeatable reports.

## 6. MVP Scope

The MVP is a Python project containing:

- Python package `semantic_kv`.
- Prompt segmenter that splits structured requests into typed candidate blocks.
- Semantic block classifier using rules first, with optional embedding/model hooks later.
- Cache simulator that models block admission, access, eviction, memory capacity, and latency estimates.
- Eviction policy interface shared by all policies.
- Baseline policies: LRU, LFU, FIFO, size-aware LRU.
- Static semantic weighting policy.
- Adaptive semantic policy with online updates.
- Synthetic workload generator for realistic multi-session workloads.
- FastAPI service for simulation and cache operations.
- Metrics export in Prometheus-compatible format.
- Experiment runner that compares policies and writes reports.
- Basic dashboard or generated report with plots for hit rate, TTFT estimates, evictions, and semantic weights.

## 7. System Architecture

```text
                      +----------------------------+
                      | Client / Request Generator |
                      +-------------+--------------+
                                    |
                                    v
                         +----------+----------+
                         | FastAPI Gateway     |
                         | apps/api            |
                         +----------+----------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
      +----------+-----------+              +----------+-----------+
      | Prompt Segmentation  |              | Experiment Runner    |
      | semantic_kv.segment  |              | semantic_kv.experiments
      +----------+-----------+              +----------+-----------+
                 |                                     |
                 v                                     v
      +----------+-----------+              +----------+-----------+
      | Semantic Classifier  |              | Workload Generator   |
      | rules / future model |              | synthetic sessions   |
      +----------+-----------+              +----------+-----------+
                 |
                 v
      +----------+-----------+
      | Cache Admission      |
      | hash, dedupe, size   |
      +----------+-----------+
                 |
                 v
      +----------+-----------+        +----------------------+
      | Policy Engine        |<------>| Cache State Store    |
      | LRU/LFU/FIFO/adapt   |        | memory / Redis opt.  |
      +----------+-----------+        +----------------------+
                 |
                 v
      +----------+-----------+
      | Metrics Collector    |
      | Prometheus / reports |
      +----------+-----------+
                 |
                 v
      +----------+-----------+
      | Optional Backend     |
      | mock / vLLM / SGLang |
      +----------------------+
```

Optional infrastructure:

- Redis for shared cache metadata.
- PostgreSQL for experiment logs and run metadata.
- Prometheus for metrics scraping.
- Grafana for dashboards.
- LLM backend adapter for vLLM, SGLang, Ray Serve, or a mock backend.

## 8. Core Data Models

### `PromptRequest`

Purpose: Represents a single LLM request before segmentation.

Fields:

- `request_id: str` unique request identifier.
- `session_id: str | None` conversation or workload session identifier.
- `tenant_id: str | None` optional organization/user namespace.
- `timestamp_ms: int` event time for simulation.
- `messages: list[dict]` chat-style messages or structured prompt parts.
- `raw_prompt: str | None` flat prompt representation when messages are unavailable.
- `metadata: dict[str, Any]` workload labels, route, model, or experiment tags.
- `max_tokens: int | None` requested generation length.
- `model_name: str | None` target model or simulator profile.

### `PromptBlock`

Purpose: Represents a semantically classified prompt segment before cache admission.

Fields:

- `block_id: str` stable hash of normalized content and semantic metadata.
- `request_id: str` parent request.
- `session_id: str | None` parent session.
- `position: int` block order in prompt.
- `text: str` block content or synthetic text placeholder.
- `token_count: int` estimated token length.
- `semantic_type: SemanticType` classifier output.
- `source: str | None` message role, retriever id, tool name, or template id.
- `content_hash: str` hash for cache matching.
- `is_private: bool` whether the block should be excluded from logs.
- `metadata: dict[str, Any]` classifier confidence, document id, turn index, etc.

### `CacheBlock`

Purpose: Represents an admitted cache entry with policy state.

Fields:

- `cache_key: str` unique cache key.
- `content_hash: str` normalized content identity.
- `semantic_type: SemanticType` semantic class.
- `token_count: int` memory cost proxy.
- `memory_bytes_estimate: int` estimated KV memory footprint.
- `created_at_ms: int` admission time.
- `last_accessed_at_ms: int` latest hit or access.
- `access_count: int` observed reuse count.
- `session_ids: set[str]` sessions that used the block.
- `tenant_id: str | None` namespace.
- `score: float` latest policy score.
- `metadata: dict[str, Any]` policy-specific state.

### `CacheEvent`

Purpose: Immutable event emitted for admissions, hits, misses, and evictions.

Fields:

- `event_id: str` unique event identifier.
- `timestamp_ms: int` event time.
- `request_id: str | None` associated request.
- `event_type: Literal["hit", "miss", "admit", "evict", "reject"]`.
- `cache_key: str | None` affected cache key.
- `semantic_type: SemanticType | None` affected semantic class.
- `token_count: int` tokens involved.
- `policy_name: str` active policy.
- `reason: str | None` eviction/admission reason.
- `estimated_latency_delta_ms: float` estimated TTFT impact.
- `metadata: dict[str, Any]` run id, workload, score breakdown.

### `CacheDecision`

Purpose: Explains policy decisions for admission or eviction.

Fields:

- `decision_id: str` unique decision id.
- `request_id: str | None` associated request.
- `policy_name: str` policy that made the decision.
- `admit_keys: list[str]` blocks admitted.
- `evict_keys: list[str]` blocks evicted.
- `reject_keys: list[str]` blocks not admitted.
- `score_by_key: dict[str, float]` total policy score.
- `score_breakdown: dict[str, dict[str, float]]` interpretable components.
- `memory_before_tokens: int` cache size before decision.
- `memory_after_tokens: int` cache size after decision.
- `reason: str` decision summary.

### `PolicyMetrics`

Purpose: Aggregated policy performance for a run or time window.

Fields:

- `policy_name: str` policy identifier.
- `requests_total: int` processed requests.
- `hits_total: int` cache hits.
- `misses_total: int` cache misses.
- `evictions_total: int` evictions.
- `hit_rate: float` hits divided by accesses.
- `tokens_saved_total: int` estimated prefill tokens avoided.
- `estimated_ttft_ms_avg: float` mean estimated TTFT.
- `estimated_ttft_ms_p50: float` p50 estimated TTFT.
- `estimated_ttft_ms_p95: float` p95 estimated TTFT.
- `memory_tokens_used_avg: float` average cache occupancy.
- `memory_utilization_avg: float` average occupancy divided by capacity.
- `eviction_regret_total: float` cumulative regret proxy.
- `hit_rate_by_type: dict[SemanticType, float]` semantic hit rates.
- `evictions_by_type: dict[SemanticType, int]` semantic eviction counts.

### `WorkloadSession`

Purpose: Represents one synthetic or replayed user/session trajectory.

Fields:

- `session_id: str` unique session id.
- `workload_type: str` workload family.
- `tenant_id: str | None` namespace.
- `start_timestamp_ms: int` session start.
- `requests: list[PromptRequest]` ordered requests.
- `shared_artifact_ids: list[str]` documents, templates, tools, or code files reused.
- `metadata: dict[str, Any]` difficulty, domain, seed, expected reuse profile.

### `ExperimentConfig`

Purpose: Defines a reproducible policy comparison run.

Fields:

- `experiment_id: str` unique run id.
- `seed: int` random seed.
- `policies: list[str]` policies to compare.
- `workload_families: list[str]` generated or replayed workloads.
- `num_sessions: int` number of sessions.
- `requests_per_session: int` request count per session.
- `cache_capacity_tokens: int` simulated capacity.
- `token_latency_ms: float` prefill cost model per token.
- `base_ttft_ms: float` fixed latency floor.
- `semantic_classifier: str` classifier variant.
- `output_dir: str` report and artifact destination.
- `metadata: dict[str, Any]` experiment tags and notes.

## 9. Eviction Policy Design

All policies implement a shared interface:

```python
class EvictionPolicy:
    name: str

    def on_access(self, block: CacheBlock, event: CacheEvent) -> None:
        """Update policy state after a cache hit or lookup."""

    def on_admit(self, block: CacheBlock, cache_state: "CacheState") -> None:
        """Initialize policy state for a newly admitted block."""

    def choose_evictions(
        self,
        cache_state: "CacheState",
        incoming_blocks: list[PromptBlock],
        capacity_tokens: int,
    ) -> CacheDecision:
        """Return blocks to evict or reject so capacity invariants hold."""

    def update_from_feedback(
        self,
        events: list[CacheEvent],
        metrics: PolicyMetrics,
    ) -> None:
        """Update adaptive state from observed cache outcomes."""
```

### Baseline Policies

- `FIFO`: Evicts blocks in admission order. Useful as a simple lower-bound baseline.
- `LRU`: Evicts the least recently accessed block. Strong general baseline for temporal locality.
- `LFU`: Evicts blocks with the lowest access count, with deterministic tie-breaking by age or recency.
- `SizeAwareLRU`: Evicts low-recency blocks while accounting for token cost, preferring to remove large stale entries when capacity pressure is high.
- `StaticSemanticWeight`: Scores blocks using fixed semantic weights plus memory cost, testing whether hand-authored semantic intuition is enough.

### Adaptive Semantic Policy

The adaptive semantic policy uses a multi-queue design by semantic type. Each semantic type has its own queue and learned weight. Candidate evictions are scored across queues using a common value function.

State tracked per block:

- Semantic type.
- Token count and memory estimate.
- Last access time.
- Access count.
- Estimated tokens saved from previous hits.
- Cross-session reuse count.
- Admission age.
- Recent regret indicators.

State tracked per semantic type:

- `semantic_weight[type]`.
- Rolling hit rate.
- Rolling eviction regret.
- Rolling tokens saved per admitted token.
- Decayed reuse interval.

Candidate value score:

```text
value(block, t) =
    semantic_weight[type]
  + alpha * log1p(access_count)
  + beta  * exp(-(t - last_accessed_at_ms) / recency_tau_ms)
  + gamma * estimated_latency_saved_ms
  + delta * cross_session_reuse_count
  - eta   * normalized_memory_cost
  - zeta  * staleness_penalty
```

Eviction chooses the lowest-value blocks until projected capacity is below the limit. For incoming blocks, admission can be rejected if their initial value is lower than the retained alternatives.

Online feedback update:

- On hit: increase the semantic type weight proportional to estimated tokens saved and reuse surprise.
- On miss: no direct penalty unless the block was previously evicted.
- On regretted eviction: increase the semantic weight and local feature contribution for the evicted block type.
- On stale retention: decrease semantic weights for block types that consume memory without hits over a rolling window.

Regret signal:

```text
regret(evicted_block) =
    estimated_latency_saved_if_present_ms
  + lambda_tokens * evicted_block.token_count
  - lambda_age * time_since_eviction_ms
```

A regretted eviction occurs when a request later accesses a block that was evicted within a configurable regret horizon. This does not require exact GPU measurement; it is a simulator feedback signal used to learn better relative priorities.

## 10. Workload Generator

### Repeated System Prompt Workload

Requests share large system prompts, developer instructions, and formatting templates. User queries are mostly unique.

Expected behavior: semantic-aware policies should retain `system` and `template` blocks. LRU can perform well if traffic is dense but may evict shared prompts during mixed workloads.

### RAG Over Shared Document Corpus

Sessions retrieve chunks from a shared document set. Some documents are hot, some are long-tail, and query text is mostly unique.

Expected behavior: adaptive policy should learn which `retrieved_context` blocks are reused and distinguish hot shared chunks from one-off retrievals.

### Multi-Turn Support Agent Workload

Sessions reuse system prompts, policy templates, customer history snippets, assistant history, and occasional tool outputs.

Expected behavior: within-session history has local value; global system and policy blocks have cross-session value. Adaptive policy should balance session-local and global reuse.

### Coding Assistant Workload

Requests include repeated repository instructions, file snippets, prior assistant edits, test failures, and user-specific commands.

Expected behavior: file context and build/test output may be reused across nearby turns. Stale assistant history should decay faster than stable repository instructions.

### Legal/Evidence Review Workload

Long prompts contain case templates, statutes, evidence excerpts, citations, and reviewer questions.

Expected behavior: static templates and repeatedly cited evidence should be retained. Large unique evidence blocks should face memory-cost penalties unless reuse appears.

### Tool-Using Agent Workload

Requests contain system prompts, tool schemas, tool outputs, planning traces represented only as synthetic/private labels, and final task prompts.

Expected behavior: tool schemas may be highly reusable, tool outputs vary by task, and synthetic/private reasoning labels should be used only in simulation metadata.

## 11. Metrics and Evaluation

### Primary Metrics

- `estimated_ttft_ms`: estimated time to first token using a configurable prefill cost model.
- `cache_hit_rate`: block hits divided by block lookups.
- `prefix_reuse_depth`: number of leading prompt tokens served from cache.
- `prefill_tokens_saved`: tokens skipped due to cache hits.
- `eviction_regret`: estimated cost of evicting blocks later requested within a regret horizon.
- `memory_utilization`: used cache tokens divided by capacity.
- `hit_rate_by_semantic_type`: per-type hit rate.
- `policy_adaptation_curve`: semantic weights and rolling hit rates over time.
- `p50/p95_latency_estimates`: latency distribution under simulated serving conditions.

### Experiment Comparisons

- Adaptive semantic policy vs LRU.
- Adaptive semantic policy vs LFU.
- Adaptive semantic policy vs FIFO.
- Adaptive semantic policy vs size-aware LRU.
- Adaptive semantic policy vs static semantic weights.
- Workload mismatch test: train/adapt on one workload phase, evaluate after traffic shifts.
- Low-memory stress test: reduce cache capacity until policies diverge.
- Long-context RAG test: increase document chunk sizes and retrieval fanout.
- Ablation: adaptive policy without semantic weights.
- Ablation: adaptive policy without regret feedback.

## 12. API Design

### `POST /v1/simulate/request`

Runs segmentation, cache lookup, admission, eviction, and metric updates for one request.

Request:

```json
{
  "request_id": "req_001",
  "session_id": "sess_42",
  "tenant_id": "tenant_a",
  "model_name": "llama-3.1-8b",
  "messages": [
    {"role": "system", "content": "You are a support assistant."},
    {"role": "user", "content": "Summarize ticket ACME-123."}
  ],
  "metadata": {"workload": "support_agent"}
}
```

Response:

```json
{
  "request_id": "req_001",
  "policy_name": "adaptive_semantic",
  "hits": 1,
  "misses": 1,
  "tokens_saved": 36,
  "estimated_ttft_ms": 148.2,
  "admitted_blocks": ["blk_user_9af"],
  "evicted_blocks": [],
  "decision_id": "dec_001"
}
```

### `POST /v1/cache/admit`

Admits explicit blocks into the simulated cache.

Request:

```json
{
  "blocks": [
    {
      "block_id": "blk_sys_001",
      "text": "You are a support assistant.",
      "token_count": 36,
      "semantic_type": "system",
      "metadata": {"template_id": "support_v1"}
    }
  ]
}
```

Response:

```json
{
  "admitted": ["blk_sys_001"],
  "rejected": [],
  "evicted": [],
  "memory_tokens_used": 36
}
```

### `POST /v1/cache/access`

Accesses one or more cache keys and updates policy state.

Request:

```json
{
  "request_id": "req_002",
  "cache_keys": ["blk_sys_001", "blk_doc_007"]
}
```

Response:

```json
{
  "hits": ["blk_sys_001"],
  "misses": ["blk_doc_007"],
  "tokens_saved": 36,
  "events": ["evt_101", "evt_102"]
}
```

### `POST /v1/experiments/run`

Runs a configured policy comparison.

Request:

```json
{
  "experiment_id": "rag_low_memory_001",
  "seed": 17,
  "policies": ["lru", "lfu", "fifo", "static_semantic", "adaptive_semantic"],
  "workload_families": ["rag_shared_corpus"],
  "num_sessions": 200,
  "requests_per_session": 8,
  "cache_capacity_tokens": 16000,
  "output_dir": "runs/rag_low_memory_001"
}
```

Response:

```json
{
  "experiment_id": "rag_low_memory_001",
  "status": "completed",
  "best_policy": "adaptive_semantic",
  "report_path": "runs/rag_low_memory_001/report.md",
  "metrics_path": "runs/rag_low_memory_001/metrics.parquet"
}
```

### `GET /v1/cache/state`

Returns current cache state, optionally filtered by semantic type or tenant.

### `GET /v1/metrics/summary`

Returns aggregate metrics for the active policy or latest experiment.

### `GET /v1/metrics/timeseries`

Returns time-series metrics for dashboarding.

### `GET /health`

Returns service health and build metadata.

Response:

```json
{
  "status": "ok",
  "service": "semantic-kv-api",
  "version": "0.1.0"
}
```

## 13. Storage and Infrastructure

MVP storage should be in-memory to keep iteration fast and deterministic. The in-memory store holds cache metadata, policy state, events, and current metrics.

Optional production-style storage:

- Redis for shared cache metadata, policy queues, and cross-process cache state.
- PostgreSQL for experiment runs, workload metadata, event logs, and report indexes.
- Parquet and CSV output for reproducible offline analysis.
- Prometheus `/metrics` endpoint for scrapeable service metrics.
- Grafana dashboard provisioned through Docker Compose.
- Docker Compose for local deployment of API, Redis, PostgreSQL, Prometheus, and Grafana.

The MVP should keep storage interfaces explicit so `InMemoryCacheStateStore` can later be swapped with `RedisCacheStateStore` without changing policy code.

## 14. Observability

### Prometheus Metrics

- `semantic_kv_cache_hits_total{policy,semantic_type,workload}`
- `semantic_kv_cache_misses_total{policy,semantic_type,workload}`
- `semantic_kv_evictions_total{policy,semantic_type,reason}`
- `semantic_kv_tokens_saved_total{policy,semantic_type,workload}`
- `semantic_kv_estimated_ttft_ms{policy,workload}`
- `semantic_kv_memory_tokens_used{policy}`
- `semantic_kv_memory_capacity_tokens{policy}`
- `semantic_kv_policy_score_by_type{policy,semantic_type}`
- `semantic_kv_eviction_regret_total{policy,semantic_type}`
- `semantic_kv_requests_total{policy,workload}`

### Dashboard Panels

- Hit rate over time by policy.
- Estimated TTFT by policy with p50 and p95.
- Eviction count by semantic type.
- Memory pressure and utilization.
- Adaptive semantic weights over time.
- Tokens saved over time.
- Prefix reuse depth distribution.
- Regretted evictions by semantic type.
- Workload mix over time.

## 15. Repo Structure

```text
semantic-kv/
  apps/
    api/
      semantic_kv_api/
        main.py
        routes/
        dependencies.py
      tests/
  semantic_kv/
    cache/
      simulator.py
      state.py
      events.py
    policies/
      base.py
      fifo.py
      lru.py
      lfu.py
      size_aware_lru.py
      static_semantic.py
      adaptive_semantic.py
    segmentation/
      segmenter.py
      classifier.py
      types.py
    workloads/
      generator.py
      rag.py
      support.py
      coding.py
      legal.py
      tools.py
    experiments/
      runner.py
      reports.py
      configs.py
    metrics/
      collector.py
      prometheus.py
      timeseries.py
    storage/
      memory.py
      redis.py
      postgres.py
  tests/
    unit/
    property/
    integration/
    regression/
  configs/
    experiments/
    dashboards/
    prometheus.yml
  scripts/
    run_experiment.py
    generate_report.py
    benchmark_smoke.py
  docs/
    PRD.md
    architecture.md
    evaluation.md
    integration-notes.md
  docker-compose.yml
  pyproject.toml
  README.md
```

## 16. Testing Strategy

- Unit tests for every eviction policy, including deterministic tie-breakers and feedback updates.
- Property tests for cache capacity invariants: used tokens must never exceed capacity after an admission decision.
- Property tests for event consistency: every hit, miss, admission, and eviction should produce valid events.
- Workload generator tests verifying reproducibility from seed and expected semantic type distributions.
- API tests for simulation, direct cache admission/access, experiment runs, metrics summary, and health checks.
- Regression tests for experiment outputs using fixed seeds and small workloads.
- Benchmark smoke tests that run all policies against at least one workload and assert output files are generated.
- Observability tests ensuring Prometheus metric names and labels remain stable.

## 17. Milestones

### Milestone 1: Simulator and Baseline Policies

- Create Python package structure.
- Implement core data models.
- Implement in-memory cache state.
- Implement `EvictionPolicy` interface.
- Implement FIFO, LRU, LFU, and size-aware LRU.
- Add deterministic unit tests and property tests.

### Milestone 2: Semantic Adaptive Policy and Workload Generator

- Implement prompt segmenter and rule-based semantic classifier.
- Implement static semantic weighting policy.
- Implement adaptive semantic policy with learned weights and regret feedback.
- Build synthetic workload generator for at least four workload families.
- Add experiment fixtures and reproducibility tests.

### Milestone 3: FastAPI Service, Metrics, and Reports

- Build FastAPI gateway with simulation, cache, experiment, state, metrics, and health endpoints.
- Add Prometheus metrics export.
- Implement experiment runner with CSV/Parquet/JSON outputs.
- Generate markdown or HTML reports with comparison plots.
- Add API and regression tests.

### Milestone 4: Optional Infrastructure and Integration Notes

- Add Docker Compose with API, Redis, PostgreSQL, Prometheus, and Grafana.
- Implement optional Redis cache metadata store.
- Implement optional PostgreSQL experiment log store.
- Add Grafana dashboard provisioning.
- Write vLLM, SGLang, and Ray Serve integration notes explaining how the middleware could map simulator decisions to real prefix-cache controls when backend APIs allow it.

## 18. Risks and Mitigations

- Simulated TTFT may not match real GPU serving.
  - Mitigation: clearly label estimates, make cost models configurable, and add backend-specific calibration notes.
- Semantic classifier may be brittle.
  - Mitigation: start with explicit structured prompt segmentation, classifier confidence, and fallback `other` labels; avoid pretending the rule-based classifier is semantic understanding.
- Adaptive policy may overfit one workload.
  - Mitigation: include workload mismatch tests, phase-shifted traffic, held-out seeds, and ablations.
- Scope may balloon if vLLM internals are modified too early.
  - Mitigation: keep MVP as simulator plus middleware; defer backend integration to documented adapters.
- Dashboard complexity may distract from the core algorithm.
  - Mitigation: start with generated reports and Prometheus metrics; add Grafana only after policies and experiments are stable.
- Memory model may be too coarse.
  - Mitigation: parameterize KV memory estimates by model layers, hidden size, dtype, heads, and block size later.
- Cache matching may be unrealistic if prompt normalization is weak.
  - Mitigation: implement stable content hashing, role-aware normalization, and explicit block boundaries.

## 19. Definition of Done

SemanticKV is complete for the MVP when:

- At least five eviction policies are implemented: FIFO, LRU, LFU, size-aware LRU, static semantic weighting, and adaptive semantic.
- At least four workload families are supported.
- The experiment runner compares policies reproducibly from fixed seeds.
- The adaptive semantic policy beats LRU on at least two workload families under memory pressure according to predeclared metrics.
- Metrics are exported through a Prometheus-compatible endpoint.
- Reports show policy comparisons, semantic hit rates, latency estimates, memory utilization, and adaptation curves.
- README explains architecture, setup, experiments, and results.
- Unit, property, API, and regression tests pass.
- Docker Compose runs the API locally.

## 20. Resume Positioning

- Built `SemanticKV`, a production-style LLM serving simulator and FastAPI middleware for adaptive semantic-aware prefix/KV cache eviction, implementing LRU/LFU/FIFO/size-aware/static/adaptive policies with reproducible benchmarking across multi-session RAG, support, coding, and tool-agent workloads.
- Designed an online cache policy that learns semantic block weights from hit/miss/eviction feedback and regret signals, improving estimated TTFT, prefix hit rate, and prefill tokens saved under constrained KV-cache memory versus standard eviction baselines.
- Added infrastructure-grade observability and experiment tooling with Prometheus metrics, optional Redis/PostgreSQL storage, Docker Compose deployment, and report generation for policy adaptation curves, p50/p95 latency estimates, memory pressure, and hit rate by semantic type.
