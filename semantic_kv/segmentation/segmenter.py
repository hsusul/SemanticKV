from __future__ import annotations

import re
from typing import Any

from semantic_kv.cache.models import PromptBlock, PromptRequest
from semantic_kv.segmentation.classifier import RuleBasedSemanticClassifier
from semantic_kv.utils.hashing import normalize_text, stable_hash


def estimate_tokens(text: str) -> int:
    return max(1, int(len(re.findall(r"\S+", text)) * 1.25))


class PromptSegmenter:
    def __init__(self, classifier: RuleBasedSemanticClassifier | None = None) -> None:
        self.classifier = classifier or RuleBasedSemanticClassifier()

    def segment(self, request: PromptRequest) -> list[PromptBlock]:
        if request.messages:
            return self._segment_messages(request)
        return self._segment_raw(request)

    def _block(
        self,
        request: PromptRequest,
        text: str,
        position: int,
        role: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptBlock:
        normalized = normalize_text(text)
        semantic_type = self.classifier.classify(text, role=role, metadata=metadata)
        content_hash = stable_hash(semantic_type.value, normalized)
        return PromptBlock(
            block_id=f"blk_{stable_hash(request.request_id, position, content_hash)}",
            request_id=request.request_id,
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            position=position,
            text=text,
            token_count=estimate_tokens(text),
            semantic_type=semantic_type,
            source=source or role,
            content_hash=content_hash,
            is_private=bool((metadata or {}).get("is_private", False)),
            metadata=metadata or {},
        )

    def _segment_messages(self, request: PromptRequest) -> list[PromptBlock]:
        blocks: list[PromptBlock] = []
        position = 0
        for message in request.messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            metadata = dict(message.get("metadata", {}))
            if not content.strip():
                continue
            for part in self._split_content(content):
                blocks.append(self._block(request, part, position, role=role, metadata=metadata))
                position += 1
        return blocks

    def _segment_raw(self, request: PromptRequest) -> list[PromptBlock]:
        raw = request.raw_prompt or ""
        blocks: list[PromptBlock] = []
        for position, part in enumerate(self._split_labeled_raw(raw)):
            blocks.append(self._block(request, part, position))
        return blocks

    def _split_content(self, content: str) -> list[str]:
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()]
        return chunks or [content]

    def _split_labeled_raw(self, raw: str) -> list[str]:
        raw = raw.strip()
        if not raw:
            return []
        pieces = re.split(r"(?im)(?=^\s*(?:SYSTEM|USER|ASSISTANT|TOOL|CONTEXT|TEMPLATE|CODE)\s*:)", raw)
        pieces = [piece.strip() for piece in pieces if piece.strip()]
        if len(pieces) > 1:
            return pieces
        return self._split_content(raw)

