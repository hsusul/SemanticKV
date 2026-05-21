from __future__ import annotations

from semantic_kv.traces.models import ServingTrace, TracePromptBlock, TraceRequest
from semantic_kv.utils.hashing import normalize_text, stable_hash


def anonymize_trace(trace: ServingTrace, drop_text: bool = True) -> ServingTrace:
    requests: list[TraceRequest] = []
    for request in trace.requests:
        blocks = [anonymize_block(block, drop_text=drop_text) for block in request.blocks]
        requests.append(request.model_copy(update={"blocks": blocks}))
    return trace.model_copy(update={"requests": requests})


def anonymize_block(block: TracePromptBlock, drop_text: bool = True) -> TracePromptBlock:
    text = block.text or block.content_hash or block.block_id or ""
    content_hash = block.content_hash or stable_hash(block.semantic_type.value, normalize_text(text), block.token_count)
    return block.model_copy(
        update={
            "content_hash": content_hash,
            "text": None if drop_text else block.text,
        }
    )

