from __future__ import annotations

from dataclasses import dataclass, field

from semantic_kv.cache.models import CacheBlock, CacheEvent


@dataclass
class CacheState:
    capacity_tokens: int
    blocks: dict[str, CacheBlock] = field(default_factory=dict)
    hash_index: dict[str, str] = field(default_factory=dict)
    events: list[CacheEvent] = field(default_factory=list)
    evicted_recent: dict[str, CacheBlock] = field(default_factory=dict)

    @property
    def used_tokens(self) -> int:
        return sum(block.token_count for block in self.blocks.values())

    def get_by_hash(self, content_hash: str) -> CacheBlock | None:
        key = self.hash_index.get(content_hash)
        if key is None:
            return None
        return self.blocks.get(key)

    def add(self, block: CacheBlock) -> None:
        self.blocks[block.cache_key] = block
        self.hash_index[block.content_hash] = block.cache_key

    def remove(self, cache_key: str) -> CacheBlock | None:
        block = self.blocks.pop(cache_key, None)
        if block is None:
            return None
        if self.hash_index.get(block.content_hash) == cache_key:
            self.hash_index.pop(block.content_hash, None)
        self.evicted_recent[block.content_hash] = block
        return block

    def append_event(self, event: CacheEvent) -> None:
        self.events.append(event)

    def snapshot(self) -> dict:
        return {
            "capacity_tokens": self.capacity_tokens,
            "used_tokens": self.used_tokens,
            "block_count": len(self.blocks),
            "blocks": [block.model_dump(mode="json") for block in self.blocks.values()],
        }

