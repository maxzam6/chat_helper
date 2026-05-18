from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class ActiveMemoryCache:
    """In-memory workspace for the currently active user's relevant memories."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.user_id: str | None = None
        self.retrieval_query = ""
        self.query_embedding: list[float] | None = None
        self.memories: list[dict[str, Any]] = []
        self.dirty_memory_ids: set[int] = set()
        self.updated_at: str | None = None

    def is_for_user(self, user_id: str | None) -> bool:
        return bool(user_id) and self.user_id == user_id

    def has_cache(self, user_id: str | None) -> bool:
        return self.is_for_user(user_id) and bool(self.memories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "retrieval_query": self.retrieval_query,
            "query_embedding": self.query_embedding,
            "memories": self.memories,
            "dirty_memory_ids": sorted(self.dirty_memory_ids),
            "updated_at": self.updated_at,
        }

    def set_cache(
        self,
        user_id: str,
        retrieval_query: str,
        memories: list[dict[str, Any]],
        query_embedding: list[float] | None = None,
    ) -> None:
        self.user_id = user_id
        self.retrieval_query = retrieval_query
        self.query_embedding = query_embedding
        self.memories = list(memories)
        self.dirty_memory_ids.clear()
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_memories(self, statuses: list[str] | None = None) -> list[dict[str, Any]]:
        if statuses is None:
            return list(self.memories)
        return [memory for memory in self.memories if memory.get("memory_status") in statuses]

    def upsert_memory(self, memory: dict[str, Any], dirty: bool = False) -> None:
        memory_id = memory.get("id")
        if memory_id is None:
            return
        for index, existing in enumerate(self.memories):
            if existing.get("id") == memory_id:
                self.memories[index] = memory
                break
        else:
            self.memories.append(memory)
        if dirty:
            self.dirty_memory_ids.add(int(memory_id))
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_dirty(self, memory_id: int) -> None:
        self.dirty_memory_ids.add(memory_id)

    def get_dirty_memories(self) -> list[dict[str, Any]]:
        dirty_ids = self.dirty_memory_ids
        return [memory for memory in self.memories if int(memory.get("id", -1)) in dirty_ids]

    def clear_dirty(self) -> None:
        self.dirty_memory_ids.clear()
