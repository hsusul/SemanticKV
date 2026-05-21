from __future__ import annotations

import hashlib
import re


def normalize_text(text: str) -> str:
    """Normalize prompt text for stable cache matching."""
    return re.sub(r"\s+", " ", text.strip())


def stable_hash(*parts: object, length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]

