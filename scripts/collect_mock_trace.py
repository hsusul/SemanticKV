#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_kv.collectors.jsonl_collector import JsonlTraceCollector
from semantic_kv.collectors.mock_backend import MockLLMBackend
from semantic_kv.traces.models import TraceReplayConfig
from semantic_kv.traces.replay import TraceReplayRunner
from semantic_kv.utils.time import now_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a serving-style trace from the mock LLM backend.")
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--output-name", default="mock_backend_trace")
    parser.add_argument("--output-dir", default="outputs/live_traces")
    parser.add_argument("--keep-raw-text", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--cache-token-budget", type=int, default=10000)
    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{now_ms()}_{args.output_name}.jsonl"
    backend = MockLLMBackend()
    collector = JsonlTraceCollector(
        output_path=output_path,
        backend_name="mock_backend",
        model_name=backend.model_name,
        drop_raw_text=not args.keep_raw_text,
    )
    try:
        for idx in range(args.requests):
            prompt = _prompt_for_idx(idx)
            with collector.trace_request(
                prompt=prompt,
                model_name=backend.model_name,
                session_id=f"mock-session-{idx // 5}",
                tenant_id="demo-tenant",
                metadata={"source": "collect_mock_trace", "request_index": idx},
            ) as span:
                response = backend.generate(prompt)
                span.set_response_text(response["text"])
                span.set_observed_output_tokens(response["output_tokens"])
                span.set_observed_prefix_hit_tokens(response["observed_prefix_hit_tokens"])
                span.set_observed_prefill_tokens(response["observed_prefill_tokens"])
                span.set_backend_metadata(response["metadata"])
    finally:
        collector.close()

    print("SemanticKV mock trace collection complete")
    print(f"Requests: {args.requests}")
    print(f"Trace path: {output_path}")
    print(f"Raw text stored: {args.keep_raw_text}")

    if args.replay:
        replay_config = TraceReplayConfig(
            trace_path=str(output_path),
            cache_token_budget=args.cache_token_budget,
            policies=["lru", "adaptive_semantic", "static_semantic"],
            output_name=f"{args.output_name}_replay",
            calibrate_from_trace=True,
        )
        result = TraceReplayRunner(replay_config).run(write_outputs=True)
        print(f"Replay output: {result['output_dir']}")
    else:
        print(
            "Replay command: "
            f"python scripts/replay_trace.py --trace {output_path} "
            "--cache-token-budget 10000 --policies lru adaptive_semantic static_semantic"
        )


def _prompt_for_idx(idx: int) -> str:
    system = "SYSTEM: You are a retrieval-grounded enterprise assistant."
    template = "TEMPLATE: answer with summary, citations, and next action."
    docs = [
        "CONTEXT: Shared policy document A. " + "Refund policy eligibility and approval workflow. " * 12,
        "CONTEXT: Shared policy document B. " + "Security review requirements and exception handling. " * 12,
        "CONTEXT: Shared policy document C. " + "Data retention controls and customer notification rules. " * 12,
    ]
    doc = docs[idx % len(docs)]
    user = f"USER: request {idx} asks about account {idx % 7} with unique details {idx * 17}."
    return "\n\n".join([system, template, doc, user])


if __name__ == "__main__":
    main()

