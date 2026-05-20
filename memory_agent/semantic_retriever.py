from __future__ import annotations

from pathlib import Path
import math
from typing import Any


class SemanticRetriever:
    """Lightweight semantic retrieval index backed by ChromaDB when available.

    SQLite is still the source of truth. This class only indexes memory text for
    semantic recall. If chromadb or sentence-transformers is not installed, the
    class falls back to a tiny in-memory lexical index so the local backend can
    still run without network installs.
    """

    DEFAULT_COLLECTION_NAME = "user_memory_bge_base_zh_v15"
    DEFAULT_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
    DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        persist_path: str | Path = "chroma_memory",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._fallback_items: dict[str, dict[str, Any]] = {}

        self._model = None
        self._collection = None
        self._available = False
        self._init_chroma()

    def add_memory(
        self,
        memory_id: int,
        user_id: str,
        content: str,
        memory_status: str,
        memory_type: str | None,
    ) -> None:
        """Add or update one memory vector in the retrieval index.

        The Agent skips discard memories before calling this method. If a later
        review changes a memory to another status, calling add_memory again with
        the same memory_id updates the metadata used by query filters.
        """
        item_id = str(memory_id)
        metadata = {
            "memory_id": memory_id,
            "user_id": user_id,
            "memory_status": memory_status,
            "memory_type": memory_type or "",
        }

        if not self._available:
            self._fallback_items[item_id] = {
                "id": item_id,
                "content": content,
                "metadata": metadata,
            }
            return

        embedding = self._embed(content)
        self._collection.upsert(
            ids=[item_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidate memory ids for one user.

        statuses defaults to ["stable"] so pending/conflict memories do not enter
        the normal reply context unless the caller explicitly asks for them.
        SQLite must be queried afterward for the final memory records.
        """
        statuses = statuses or ["stable"]
        query_text = query_text.strip()
        if not query_text:
            return []

        if not self._available:
            return self._fallback_query(user_id, query_text, top_k, statuses)

        where = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"memory_status": {"$in": statuses}},
            ]
        }
        results = self._collection.query(
            query_embeddings=[self._embed(self._build_query_text(query_text))],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )
        return self._format_chroma_results(results)

    def delete_memory(self, memory_id: int) -> None:
        """Remove one memory from the retrieval index."""
        item_id = str(memory_id)
        if not self._available:
            self._fallback_items.pop(item_id, None)
            return

        self._collection.delete(ids=[item_id])

    def embed_text(self, text: str) -> list[float] | None:
        """Return an embedding when the sentence-transformers model is available."""
        text = text.strip()
        if not text or self._model is None:
            return None
        try:
            return self._embed(text)
        except Exception:
            return None

    def cosine_similarity(
        self,
        vec1: list[float] | None,
        vec2: list[float] | None,
    ) -> float:
        """Compute cosine similarity for two vectors; return 0.0 on bad input."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except Exception:
            return

        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(self.model_name)
            client = chromadb.PersistentClient(path=str(self.persist_path))
            self._collection = client.get_or_create_collection(self.collection_name)
            self._available = True
        except Exception:
            self._model = None
            self._collection = None
            self._available = False

    def _embed(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _build_query_text(self, query_text: str) -> str:
        """Apply the BGE retrieval instruction to query text only.

        Memory documents are embedded as-is in add_memory(). The instruction is
        only prepended when embedding a user/search query for retrieval.
        """
        if not self.query_instruction:
            return query_text
        return f"{self.query_instruction}{query_text}"

    def _fallback_query(
        self,
        user_id: str,
        query_text: str,
        top_k: int,
        statuses: list[str],
    ) -> list[dict[str, Any]]:
        query_tokens = set(query_text.lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []

        for item in self._fallback_items.values():
            metadata = item["metadata"]
            if metadata["user_id"] != user_id:
                continue
            if metadata["memory_status"] not in statuses:
                continue

            content = item["content"]
            content_tokens = set(content.lower().split())
            overlap = len(query_tokens & content_tokens)
            if overlap == 0:
                continue
            score = float(overlap)
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "memory_id": item["metadata"]["memory_id"],
                "score": score,
            }
            for score, item in scored[:top_k]
        ]

    def _format_chroma_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted: list[dict[str, Any]] = []
        for metadata, distance in zip(metadatas, distances):
            formatted.append(
                {
                    "memory_id": metadata.get("memory_id"),
                    "score": 1.0 - float(distance),
                }
            )
        return formatted

    def _stable_tie_breaker(self, query_text: str, content: str) -> float:
        """Reserved helper for future explicit fallback ranking experiments."""
        return 0.0
