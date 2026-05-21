from __future__ import annotations

import re
from typing import Any

from semantic_kv.cache.models import SemanticType


CODE_RE = re.compile(r"```|^\s*(def |class |import |from |function |const |let |var )", re.MULTILINE)
LABEL_RE = re.compile(r"^\s*(SYSTEM|USER|ASSISTANT|TOOL|CONTEXT|TEMPLATE|CODE)\s*:", re.IGNORECASE)


class RuleBasedSemanticClassifier:
    def classify(self, text: str, role: str | None = None, metadata: dict[str, Any] | None = None) -> SemanticType:
        meta = metadata or {}
        explicit = str(meta.get("semantic_type", "")).lower()
        for semantic_type in SemanticType:
            if explicit == semantic_type.value:
                return semantic_type

        match = LABEL_RE.search(text)
        if match:
            label = match.group(1).lower()
            if label == "context":
                return SemanticType.RETRIEVED_CONTEXT
            if label == "tool":
                return SemanticType.TOOL_OUTPUT
            if label == "code":
                return SemanticType.CODE
            if label in {"system", "user", "assistant", "template"}:
                return SemanticType(label)

        role_l = (role or "").lower()
        if role_l in {"system", "developer"}:
            return SemanticType.SYSTEM
        if role_l == "user":
            return SemanticType.USER
        if role_l == "assistant":
            return SemanticType.ASSISTANT
        if role_l in {"tool", "function"}:
            return SemanticType.TOOL_OUTPUT

        lower = text.lower()
        if CODE_RE.search(text) or "traceback" in lower or "pytest" in lower:
            return SemanticType.CODE
        if any(marker in lower for marker in ["retrieved context", "context:", "document:", "evidence:", "citation:"]):
            return SemanticType.RETRIEVED_CONTEXT
        if any(marker in lower for marker in ["tool output", "observation:", "function result"]):
            return SemanticType.TOOL_OUTPUT
        if any(marker in lower for marker in ["{{", "}}", "{input}", "{question}", "template:"]):
            return SemanticType.TEMPLATE
        if len(text.split()) > 180:
            return SemanticType.RETRIEVED_CONTEXT
        return SemanticType.OTHER
