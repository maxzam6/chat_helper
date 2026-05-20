from __future__ import annotations

import json
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

        The current schema does not need old-database migration, so the table
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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        source_type TEXT,
                        source_summary TEXT,
                        last_evidence TEXT
                    )
                    """
                )
                # Legacy summary-style working memory table.
                # New GraphMemoryAgent uses working_memory_observations instead.
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
        self.ensure_schema_migrations()

    def ensure_schema_migrations(self) -> None:
        """Add columns introduced after the first local schema version."""
        required_columns = {
            "updated_at": "TIMESTAMP",
            "source_type": "TEXT",
            "source_summary": "TEXT",
            "last_evidence": "TEXT",
        }
        with closing(self._connect()) as conn:
            with conn:
                existing_columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(user_memory)").fetchall()
                }
                for column_name, column_type in required_columns.items():
                    if column_name not in existing_columns:
                        conn.execute(
                            f"ALTER TABLE user_memory ADD COLUMN {column_name} {column_type}"
                        )
                conn.execute(
                    """
                    UPDATE user_memory
                    SET updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)
                    WHERE updated_at IS NULL
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        me_id TEXT DEFAULT 'default',
                        user_id TEXT DEFAULT 'global',
                        last_intent TEXT,
                        last_user_input TEXT,
                        last_input_summary TEXT,
                        last_reply TEXT,
                        last_analysis TEXT,
                        last_chat_context TEXT,
                        last_active_user_id TEXT,
                        last_retrieval_query TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(me_id, user_id)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS working_memory_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        confidence REAL DEFAULT 0.8,
                        age INTEGER DEFAULT 0,
                        ttl INTEGER DEFAULT 5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def get_user_memory(self, user_id: str) -> list[str]:
        """Return only stable long-term memory for normal model context."""
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
                    SELECT id, user_id, memory_type, content, confidence,
                           memory_status, created_at, updated_at, source_type,
                           source_summary, last_evidence
                    FROM user_memory
                    WHERE id = ?
                    """,
                    (memory_id,),
                ).fetchone()
        return dict(row) if row else None

    def get_memory_records(self, memory_ids: list[int]) -> list[dict[str, Any]]:
        """Return full memory records for ids recalled by the vector index.

        ChromaDB/SemanticRetriever is only an index. Before sending relevant
        memories to the model, Agent calls this method so the final content comes
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
                    SELECT id, user_id, memory_type, content, confidence,
                           memory_status, created_at, updated_at, source_type,
                           source_summary, last_evidence
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
        source_type: str | None = None,
        source_summary: str | None = None,
        last_evidence: str | None = None,
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
                        memory_status,
                        updated_at,
                        source_type,
                        source_summary,
                        last_evidence
                    )
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
                    """,
                    (
                        user_id,
                        memory_type,
                        content,
                        confidence,
                        memory_status,
                        source_type,
                        source_summary,
                        last_evidence,
                    ),
                )
                return int(cursor.lastrowid)

    def find_duplicate_memory(
        self,
        user_id: str,
        memory_type: str | None,
        content: str,
        statuses: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return an existing active memory with the same user/type/content.

        This is an exact duplicate guard. It intentionally does not try semantic
        merging; near-duplicate consolidation should be handled by a later review
        or merge flow.
        """
        content = content.strip()
        if not user_id or not content:
            return None
        statuses = statuses or ["stable", "pending", "conflict"]
        for status in statuses:
            self._validate_memory_status(status)

        placeholders = ", ".join("?" for _ in statuses)
        with closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    f"""
                    SELECT id, user_id, memory_type, content, confidence,
                           memory_status, created_at, updated_at, source_type,
                           source_summary, last_evidence
                    FROM user_memory
                    WHERE user_id = ?
                      AND COALESCE(memory_type, '') = COALESCE(?, '')
                      AND TRIM(content) = ?
                      AND memory_status IN ({placeholders})
                    ORDER BY updated_at DESC, id DESC
                    LIMIT 1
                    """,
                    (user_id, memory_type, content, *statuses),
                ).fetchone()
        return dict(row) if row else None

    def update_memory_status(self, memory_id: int, new_status: str) -> bool:
        """Update a memory status directly."""
        self._validate_memory_status(new_status)

        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE user_memory
                    SET memory_status = ?,
                        updated_at = CURRENT_TIMESTAMP
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
                        memory_status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_confidence, new_status, memory_id),
                )

        if cursor.rowcount == 0:
            return None
        return new_status

    def update_memory_review(
        self,
        memory_id: int,
        confidence: float,
        memory_status: str,
    ) -> bool:
        """Update confidence/status for a reviewed memory using an explicit status."""
        self._validate_memory_status(memory_status)

        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE user_memory
                    SET confidence = ?,
                        memory_status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (confidence, memory_status, memory_id),
                )
                return cursor.rowcount > 0

    def get_working_memory(self, user_id: str) -> dict[str, Any] | None:
        """Legacy summary-style working memory getter.

        New GraphMemoryAgent should use get_working_memory_observations().
        Kept for old MemoryAgent compatibility.
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
        """Legacy summary-style working memory upsert.

        New GraphMemoryAgent should use update_working_memory_observations().
        Kept for old MemoryAgent compatibility.
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

    def get_working_memory_observations(self, user_id: str) -> list[dict[str, Any]]:
        """Return active short-term observations for one user.

        These rows are a queue, not a summary. The model generates observation text;
        Python only manages age, ttl, ordering, and max size.
        """
        with closing(self._connect()) as conn:
            with conn:
                rows = conn.execute(
                    """
                    SELECT id, user_id, content, confidence, age, ttl, created_at, updated_at
                    FROM working_memory_observations
                    WHERE user_id = ?
                    ORDER BY confidence DESC, updated_at DESC, id DESC
                    """,
                    (user_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_working_memory_observations(
        self,
        user_id: str,
        observations: list[dict[str, Any]],
        max_items: int = 8,
        default_ttl: int = 5,
    ) -> list[dict[str, Any]]:
        """Age existing observations, insert new ones, trim, and return active rows."""
        if not user_id:
            raise ValueError("user_id is required")

        cleaned_observations = [
            observation
            for observation in observations
            if str(observation.get("content", "")).strip()
        ]

        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE working_memory_observations
                    SET age = age + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
                conn.execute(
                    """
                    DELETE FROM working_memory_observations
                    WHERE user_id = ?
                      AND age >= ttl
                    """,
                    (user_id,),
                )
                for observation in cleaned_observations:
                    content = str(observation.get("content", "")).strip()
                    confidence = float(observation.get("confidence", 0.8))
                    ttl = int(observation.get("ttl", default_ttl))
                    conn.execute(
                        """
                        INSERT INTO working_memory_observations (
                            user_id,
                            content,
                            confidence,
                            age,
                            ttl,
                            updated_at
                        )
                        VALUES (?, ?, ?, 0, ?, CURRENT_TIMESTAMP)
                        """,
                        (user_id, content, confidence, ttl),
                    )

                rows_to_keep = conn.execute(
                    """
                    SELECT id
                    FROM working_memory_observations
                    WHERE user_id = ?
                    ORDER BY confidence DESC, updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (user_id, max_items),
                ).fetchall()
                keep_ids = [int(row["id"]) for row in rows_to_keep]
                if keep_ids:
                    placeholders = ", ".join("?" for _ in keep_ids)
                    conn.execute(
                        f"""
                        DELETE FROM working_memory_observations
                        WHERE user_id = ?
                          AND id NOT IN ({placeholders})
                        """,
                        (user_id, *keep_ids),
                    )
                else:
                    conn.execute(
                        """
                        DELETE FROM working_memory_observations
                        WHERE user_id = ?
                        """,
                        (user_id,),
                    )

        return self.get_working_memory_observations(user_id)

    def clear_working_memory_observations(self, user_id: str) -> None:
        """Clear short-term observations for one user."""
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    "DELETE FROM working_memory_observations WHERE user_id = ?",
                    (user_id,),
                )

    def get_session_state(
        self,
        me_id: str = "default",
        user_id: str = "global",
    ) -> dict[str, Any] | None:
        """Return saved session state for me_id/user_id."""
        with closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    """
                    SELECT me_id, user_id, last_intent, last_user_input,
                           last_input_summary, last_reply, last_analysis,
                           last_chat_context, last_active_user_id,
                           last_retrieval_query, updated_at
                    FROM session_state
                    WHERE me_id = ?
                      AND user_id = ?
                    """,
                    (me_id, user_id),
                ).fetchone()
        if not row:
            return None

        result = dict(row)
        for key in ("last_reply", "last_analysis", "last_chat_context"):
            result[key] = self._loads_json_field(result.get(key))
        return result

    def save_session_state(
        self,
        me_id: str,
        user_id: str,
        data: dict[str, Any],
    ) -> None:
        """Upsert session state used by revise and continuity flows."""
        last_reply = self._dumps_json_field(data.get("last_reply"))
        last_analysis = self._dumps_json_field(data.get("last_analysis"))
        last_chat_context = self._dumps_json_field(data.get("last_chat_context"))

        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO session_state (
                        me_id,
                        user_id,
                        last_intent,
                        last_user_input,
                        last_input_summary,
                        last_reply,
                        last_analysis,
                        last_chat_context,
                        last_active_user_id,
                        last_retrieval_query,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(me_id, user_id) DO UPDATE SET
                        last_intent = excluded.last_intent,
                        last_user_input = excluded.last_user_input,
                        last_input_summary = excluded.last_input_summary,
                        last_reply = excluded.last_reply,
                        last_analysis = excluded.last_analysis,
                        last_chat_context = excluded.last_chat_context,
                        last_active_user_id = excluded.last_active_user_id,
                        last_retrieval_query = excluded.last_retrieval_query,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        me_id or "default",
                        user_id or "global",
                        data.get("last_intent"),
                        data.get("last_user_input"),
                        data.get("last_input_summary"),
                        last_reply,
                        last_analysis,
                        last_chat_context,
                        data.get("last_active_user_id"),
                        data.get("last_retrieval_query"),
                    ),
                )

    def _get_memory_rows_by_status(self, user_id: str, memory_status: str) -> list[sqlite3.Row]:
        self._validate_memory_status(memory_status)

        with closing(self._connect()) as conn:
            with conn:
                rows = conn.execute(
                    """
                    SELECT id, user_id, memory_type, content, confidence,
                           memory_status, created_at, updated_at, source_type,
                           source_summary, last_evidence
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

    def _dumps_json_field(self, value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def _loads_json_field(self, value: Any) -> Any:
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None


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


def get_working_memory_observations(
    user_id: str,
    db_path: str | Path = "memory.db",
) -> list[dict[str, Any]]:
    return MemoryStore(db_path).get_working_memory_observations(user_id)


def update_working_memory_observations(
    user_id: str,
    observations: list[dict[str, Any]],
    max_items: int = 8,
    default_ttl: int = 5,
    db_path: str | Path = "memory.db",
) -> list[dict[str, Any]]:
    return MemoryStore(db_path).update_working_memory_observations(
        user_id,
        observations,
        max_items,
        default_ttl,
    )


def get_session_state(
    me_id: str = "default",
    user_id: str = "global",
    db_path: str | Path = "memory.db",
) -> dict[str, Any] | None:
    return MemoryStore(db_path).get_session_state(me_id, user_id)


def save_session_state(
    me_id: str,
    user_id: str,
    data: dict[str, Any],
    db_path: str | Path = "memory.db",
) -> None:
    MemoryStore(db_path).save_session_state(me_id, user_id, data)


def save_memory(
    user_id: str,
    memory_type: str | None,
    content: str,
    confidence: float = 0.8,
    memory_status: str = "stable",
    source_type: str | None = None,
    source_summary: str | None = None,
    last_evidence: str | None = None,
    db_path: str | Path = "memory.db",
) -> int:
    return MemoryStore(db_path).save_memory(
        user_id,
        memory_type,
        content,
        confidence,
        memory_status,
        source_type,
        source_summary,
        last_evidence,
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


def update_memory_review(
    memory_id: int,
    confidence: float,
    memory_status: str,
    db_path: str | Path = "memory.db",
) -> bool:
    return MemoryStore(db_path).update_memory_review(
        memory_id,
        confidence,
        memory_status,
    )
