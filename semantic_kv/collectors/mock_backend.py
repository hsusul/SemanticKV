from __future__ import annotations

from dataclasses import dataclass, field

from semantic_kv.segmentation.segmenter import estimate_tokens
from semantic_kv.utils.hashing import normalize_text, stable_hash


@dataclass
class MockLLMBackend:
    model_name: str = "mock-llm"
    base_ttft_ms: float = 35.0
    ms_per_prefill_token: float = 0.04
    ms_per_output_token: float = 0.75
    cache_hashes: set[str] = field(default_factory=set)

    def generate(self, prompt: str) -> dict:
        normalized = normalize_text(prompt)
        prompt_hash = stable_hash(normalized)
        prompt_tokens = estimate_tokens(prompt)
        observed_prefix_hit_tokens = int(prompt_tokens * 0.65) if prompt_hash in self.cache_hashes else 0
        observed_prefill_tokens = max(0, prompt_tokens - observed_prefix_hit_tokens)
        output_tokens = 12 + (int(stable_hash(prompt_hash, length=4), 16) % 24)
        observed_ttft_ms = self.base_ttft_ms + observed_prefill_tokens * self.ms_per_prefill_token
        observed_total_latency_ms = observed_ttft_ms + output_tokens * self.ms_per_output_token
        self.cache_hashes.add(prompt_hash)
        return {
            "text": f"mock response {stable_hash(prompt_hash, output_tokens, length=10)}",
            "output_tokens": output_tokens,
            "observed_prefix_hit_tokens": observed_prefix_hit_tokens,
            "observed_prefill_tokens": observed_prefill_tokens,
            "metadata": {
                "model_name": self.model_name,
                "prompt_tokens": prompt_tokens,
                "observed_ttft_ms": round(observed_ttft_ms, 4),
                "observed_total_latency_ms": round(observed_total_latency_ms, 4),
                "observed_prefix_hit_tokens": observed_prefix_hit_tokens,
                "observed_prefill_tokens": observed_prefill_tokens,
            },
        }

