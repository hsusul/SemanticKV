# SemanticKV Architecture

SemanticKV is a simulator-first LLM serving infrastructure project. It does not store real KV tensors and does not claim GPU-accurate latency. It models prefix-cache metadata, memory pressure, semantic prompt block reuse, and policy decisions so eviction algorithms can be compared reproducibly.

```text
PromptRequest
  -> PromptSegmenter
  -> RuleBasedSemanticClassifier
  -> PrefixCacheSimulator
  -> EvictionPolicy
  -> CacheState + MetricsCollector
  -> ExperimentRunner / FastAPI API / Reports
```

The simulator uses content hashes for block identity, token counts as the cache capacity unit, and a configurable TTFT estimate:

```text
estimated_ttft_ms = base_ttft_ms + uncached_prompt_tokens * token_latency_ms
```

Implemented policies:

- FIFO
- LRU
- LFU
- Size-aware LRU
- Static semantic weighting
- Adaptive semantic weighting with online regret feedback

The adaptive policy scores retained blocks using semantic weight, access frequency, recency decay, estimated latency saved, and token cost. If a previously evicted block is requested again, the simulator emits a regret event and the policy increases that block type's semantic weight.

The FastAPI app wraps a long-lived in-memory simulator for interactive requests and creates fresh simulators for experiment runs. Optional Redis/Postgres/Grafana integration is intentionally deferred so tests and local execution require no external services.

