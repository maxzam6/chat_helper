from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from memory_agent.semantic_retriever import SemanticRetriever


INDEXED_STATUSES = {"stable", "pending", "conflict"}


def main() -> int:
    load_dotenv()
    db_path = Path(os.getenv("MEMORY_DB_PATH", "memory.db"))
    retriever = SemanticRetriever(
        persist_path=os.getenv("CHROMA_DB_PATH", "chroma_memory"),
        collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            SemanticRetriever.DEFAULT_COLLECTION_NAME,
        ),
        model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            SemanticRetriever.DEFAULT_MODEL_NAME,
        ),
        query_instruction=os.getenv(
            "EMBEDDING_QUERY_INSTRUCTION",
            SemanticRetriever.DEFAULT_QUERY_INSTRUCTION,
        ),
    )

    if not retriever._available:
        raise RuntimeError("SemanticRetriever is not available; check embedding dependencies/model download.")

    rows = _load_memory_rows(db_path)
    indexed = 0
    skipped = 0
    for row in rows:
        if row["memory_status"] not in INDEXED_STATUSES:
            skipped += 1
            continue
        retriever.add_memory(
            memory_id=int(row["id"]),
            user_id=row["user_id"],
            content=row["content"],
            memory_status=row["memory_status"],
            memory_type=row["memory_type"],
        )
        indexed += 1

    print(f"indexed={indexed}")
    print(f"skipped={skipped}")
    print(f"collection={retriever.collection_name}")
    print(f"model={retriever.model_name}")
    return 0


def _load_memory_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT id, user_id, memory_type, content, memory_status
            FROM user_memory
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
