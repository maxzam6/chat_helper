from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


VALID_MEMORY_STATUSES = {"stable", "pending", "conflict", "discard"}


def classify_memory_status(confidence: float, has_conflict: bool = False) -> str:
    """Classify a memory by confidence and conflict signal."""
    if has_conflict:
        return "conflict"
    if confidence >= 0.7:
        return "stable"
    if confidence >= 0.5:
        return "pending"
    return "discard"


class MemoryStore:
    """SQLite storage layer for long-term memory and working memory.

    SQLite remains the source of truth. Vector indexes such as ChromaDB are only
    retrieval indexes and can be rebuilt from this database later.
    """

    def __init__(self, db_path: str | Path = "memory.db") -> None:
        self.db_path = Path(db_path)

    def init_db(self) -> None:
        """Create the current database schema.

        The current stage does not need old-database migration, so the table
        structure is declared directly here.
        """
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        memory_type TEXT,
                        content TEXT NOT NULL,
                        confidence REAL DEFAULT 0.8,
                        memory_status TEXT DEFAULT 'stable',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS working_memory (
                        user_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        confidence REAL DEFAULT 0.8,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def get_user_memory(self, user_id: str) -> list[str]:
        """Return only stable long-term memory for normal Dify context."""
        return [row["content"] for row in self._get_memory_rows_by_status(user_id, "stable")]

    def get_pending_memory(self, user_id: str) -> list[str]:
        """Return pending memory for review flows."""
        return [row["content"] for row in self._get_memory_rows_by_status(user_id, "pending")]

    def get_conflict_memory(self, user_id: str) -> list[str]:
        """Return conflict memory for conflict-resolution flows."""
        return [row["content"] for row in self._get_memory_rows_by_status(user_id, "conflict")]

    def get_memory_record(self, memory_id: int) -> dict[str, Any] | None:
        """Return one memory row by id, including status and confidence."""
        with closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    """
                    SELECT id, user_id, memory_type, content, confidence, memory_status, created_at
                    FROM user_memory
                    WHERE id = ?
                    """,
                    (memory_id,),
                ).fetchone()
        return dict(row) if row else None

    def get_memory_records(self, memory_ids: list[int]) -> list[dict[str, Any]]:
        """Return full memory records for ids recalled by the vector index.

        ChromaDB/SemanticRetriever is only an index. Before sending relevant
        memories to Dify, Agent calls this method so the final content comes
        from SQLite, which is the source of truth.
        """
        if not memory_ids:
            return []

        unique_ids = list(dict.fromkeys(memory_ids))
        placeholders = ", ".join("?" for _ in unique_ids)
        with closing(self._connect()) as conn:
            with conn:
                rows = conn.execute(
                    f"""
                    SELECT id, user_id, memory_type, content, confidence, memory_status, created_at
                    FROM user_memory
                    WHERE id IN ({placeholders})
                    """,
                    tuple(unique_ids),
                ).fetchall()

        rows_by_id = {int(row["id"]): dict(row) for row in rows}
        return [rows_by_id[memory_id] for memory_id in unique_ids if memory_id in rows_by_id]

    def save_memory(
        self,
        user_id: str,
        memory_type: str | None,
        content: str,
        confidence: float = 0.8,
        memory_status: str = "stable",
    ) -> int:
        """Insert one long-term memory row and return its SQLite id."""
        content = content.strip()
        self._validate_memory_input(user_id, content, memory_status)

        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO user_memory (
                        user_id,
                        memory_type,
                        content,
                        confidence,
                        memory_status
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, memory_type, content, confidence, memory_status),
                )
                return int(cursor.lastrowid)

    def update_memory_status(self, memory_id: int, new_status: str) -> bool:
        """Update a memory status directly."""
        self._validate_memory_status(new_status)

        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE user_memory
                    SET memory_status = ?
                    WHERE id = ?
                    """,
                    (new_status, memory_id),
                )
                return cursor.rowcount > 0

    def review_memory_status(
        self,
        memory_id: int,
        new_confidence: float,
        has_conflict: bool = False,
    ) -> str | None:
        """Review an existing memory and update confidence/status together."""
        new_status = classify_memory_status(new_confidence, has_conflict)

        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE user_memory
                    SET confidence = ?,
                        memory_status = ?
                    WHERE id = ?
                    """,
                    (new_confidence, new_status, memory_id),
                )

        if cursor.rowcount == 0:
            return None
        return new_status

    def get_working_memory(self, user_id: str) -> dict[str, Any] | None:
        """Return the user's current working memory, or None if missing.

        Python only stores and retrieves this value. Dify is responsible for
        deciding what the working memory content should be.
        """
        with closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    """
                    SELECT user_id, content, confidence, updated_at
                    FROM working_memory
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
        return dict(row) if row else None

    def replace_working_memory(
        self,
        user_id: str,
        content: str,
        confidence: float = 0.8,
    ) -> None:
        """Replace the user's working memory with Dify's latest summary.

        This is an upsert: it inserts the row when missing and overwrites content,
        confidence, and updated_at when the user already has working memory.
        """
        content = content.strip()
        if not user_id:
            raise ValueError("user_id is required")
        if not content:
            raise ValueError("content is required")

        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO working_memory (user_id, content, confidence, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        content = excluded.content,
                        confidence = excluded.confidence,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, content, confidence),
                )

    def _get_memory_rows_by_status(self, user_id: str, memory_status: str) -> list[sqlite3.Row]:
        self._validate_memory_status(memory_status)

        with closing(self._connect()) as conn:
            with conn:
                rows = conn.execute(
                    """
                    SELECT id, user_id, memory_type, content, confidence, memory_status, created_at
                    FROM user_memory
                    WHERE user_id = ?
                      AND memory_status = ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (user_id, memory_status),
                ).fetchall()
        return rows

    def _validate_memory_input(
        self,
        user_id: str,
        content: str,
        memory_status: str,
    ) -> None:
        if not user_id:
            raise ValueError("user_id is required")
        if not content:
            raise ValueError("content is required")
        self._validate_memory_status(memory_status)

    def _validate_memory_status(self, memory_status: str) -> None:
        if memory_status not in VALID_MEMORY_STATUSES:
            allowed = ", ".join(sorted(VALID_MEMORY_STATUSES))
            raise ValueError(f"invalid memory_status: {memory_status}; allowed: {allowed}")

    def _connect(self) -> sqlite3.Connection:
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def init_db(db_path: str | Path = "memory.db") -> None:
    MemoryStore(db_path).init_db()


def get_user_memory(user_id: str, db_path: str | Path = "memory.db") -> list[str]:
    return MemoryStore(db_path).get_user_memory(user_id)


def get_pending_memory(user_id: str, db_path: str | Path = "memory.db") -> list[str]:
    return MemoryStore(db_path).get_pending_memory(user_id)


def get_conflict_memory(user_id: str, db_path: str | Path = "memory.db") -> list[str]:
    return MemoryStore(db_path).get_conflict_memory(user_id)


def get_working_memory(
    user_id: str,
    db_path: str | Path = "memory.db",
) -> dict[str, Any] | None:
    return MemoryStore(db_path).get_working_memory(user_id)


def replace_working_memory(
    user_id: str,
    content: str,
    confidence: float = 0.8,
    db_path: str | Path = "memory.db",
) -> None:
    MemoryStore(db_path).replace_working_memory(user_id, content, confidence)


def save_memory(
    user_id: str,
    memory_type: str | None,
    content: str,
    confidence: float = 0.8,
    memory_status: str = "stable",
    db_path: str | Path = "memory.db",
) -> int:
    return MemoryStore(db_path).save_memory(
        user_id,
        memory_type,
        content,
        confidence,
        memory_status,
    )


def update_memory_status(
    memory_id: int,
    new_status: str,
    db_path: str | Path = "memory.db",
) -> bool:
    return MemoryStore(db_path).update_memory_status(memory_id, new_status)


def review_memory_status(
    memory_id: int,
    new_confidence: float,
    has_conflict: bool = False,
    db_path: str | Path = "memory.db",
) -> str | None:
    return MemoryStore(db_path).review_memory_status(
        memory_id,
        new_confidence,
        has_conflict,
    )
