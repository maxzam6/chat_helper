from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class ActiveMemoryCache:
    """In-memory workspace for the currently active user's relevant memories.

    dirty_memory_ids records cache entries changed by the learning step but not
    yet synchronized to SQLite/SemanticRetriever. sync_dirty_memory performs the
    durable writeback and then clears dirty markers.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.user_id: str | None = None
        self.retrieval_query = ""
        self.query_embedding: list[float] | None = None
        self.memories: list[dict[str, Any]] = []
        self.dirty_memory_ids: set[int | str] = set()
        self._temp_id_counter = 0
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
            "dirty_memory_ids": sorted(str(memory_id) for memory_id in self.dirty_memory_ids),
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
            self.add_pending_memory(memory)
            return
        for index, existing in enumerate(self.memories):
            if existing.get("id") == memory_id:
                self.memories[index] = memory
                break
        else:
            self.memories.append(memory)
        if dirty:
            self.dirty_memory_ids.add(memory_id)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_pending_memory(self, memory: dict[str, Any]) -> str:
        self._temp_id_counter += 1
        temp_id = f"tmp_{self._temp_id_counter}"
        new_memory = dict(memory)
        new_memory["id"] = temp_id
        new_memory["_is_new"] = True
        self.memories.append(new_memory)
        self.dirty_memory_ids.add(temp_id)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return temp_id

    def mark_dirty(self, memory_id: int | str) -> None:
        self.dirty_memory_ids.add(memory_id)

    def get_dirty_memories(self) -> list[dict[str, Any]]:
        dirty_ids = self.dirty_memory_ids
        return [memory for memory in self.memories if memory.get("id") in dirty_ids]

    def remove_memory(self, memory_id: int | str) -> None:
        self.memories = [
            memory for memory in self.memories
            if memory.get("id") != memory_id
        ]
        self.dirty_memory_ids.discard(memory_id)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def replace_memory_id(
        self,
        temp_id: str,
        real_id: int,
        real_record: dict[str, Any],
    ) -> None:
        replacement = dict(real_record)
        replacement["id"] = real_id
        replacement.pop("_is_new", None)
        for index, memory in enumerate(self.memories):
            if memory.get("id") == temp_id:
                self.memories[index] = replacement
                break
        else:
            self.memories.append(replacement)
        self.dirty_memory_ids.discard(temp_id)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def clear_dirty(self) -> None:
        self.dirty_memory_ids.clear()
