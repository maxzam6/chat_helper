from __future__ import annotations

from typing import Any

from .dify_client import DifyClient
from .input_filter import InputFilter
from .memory_store import MemoryStore, classify_memory_status
from .models import (
    build_dify_inputs,
    extract_memory_review,
    extract_memory_reviews,
    extract_memory_updates,
    extract_reply,
    extract_retrieval_query,
    extract_updated_working_memory,
)
from .semantic_retriever import SemanticRetriever


class MemoryAgent:
    """Agent workflow coordinator.

    Python does not do emotion analysis, reply generation, or working-memory
    summarization. It only coordinates filtering, storage, state rules, semantic
    retrieval, and Dify calls.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        dify_client: DifyClient,
        input_filter: InputFilter | None = None,
        semantic_retriever: SemanticRetriever | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.dify_client = dify_client
        self.input_filter = input_filter or InputFilter()
        self.semantic_retriever = semantic_retriever or SemanticRetriever()

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process one chat input through the two-stage Dify workflow."""
        self._validate_payload(payload)
        self.memory_store.init_db()

        filter_result = self.input_filter.check(payload)
        if not filter_result.should_process:
            return {
                "status": "skipped",
                "reason": filter_result.reason,
                "memory_saved": False,
            }

        user_id = payload["user_id"]
        working_memory = self.memory_store.get_working_memory(user_id)

        # Stage 1: Dify receives current chat + working memory and produces a
        # retrieval query. Python does not decide what should be searched.
        retrieval_inputs = build_dify_inputs(
            payload=payload,
            working_memory=working_memory,
            stage="retrieval_query",
        )
        retrieval_output = self.dify_client.run_workflow(retrieval_inputs)
        retrieval_query = extract_retrieval_query(retrieval_output)

        # Python uses the retrieval query only to recall candidate memory ids.
        # Full records sent to Dify are loaded from SQLite, the source of truth.
        semantic_results = self.semantic_retriever.query(
            user_id=user_id,
            query_text=retrieval_query,
            top_k=5,
            statuses=["stable"],
        )
        memory_ids = [
            int(result["memory_id"])
            for result in semantic_results
            if result.get("memory_id") is not None
        ]
        relevant_memories = [
            record
            for record in self.memory_store.get_memory_records(memory_ids)
            if record["user_id"] == user_id and record["memory_status"] == "stable"
        ]

        # Stage 2: Dify receives chat + working memory + semantically recalled
        # memories, then handles reply, learning, reviews, and working memory.
        learning_inputs = build_dify_inputs(
            payload=payload,
            working_memory=working_memory,
            relevant_memories=relevant_memories,
            stage="learning",
        )
        learning_output = self.dify_client.run_workflow(learning_inputs)

        memory_updates = extract_memory_updates(learning_output)
        saved_memory_ids = self._apply_memory_updates(user_id, memory_updates)

        memory_reviews = extract_memory_reviews(learning_output)
        reviewed_memories = self._apply_memory_reviews(memory_reviews)

        reply = extract_reply(learning_output)
        updated_working_memory = extract_updated_working_memory(learning_output)
        working_memory_saved = self._apply_working_memory_update(
            user_id,
            updated_working_memory,
        )

        return {
            "status": "processed",
            "retrieval_inputs": retrieval_inputs,
            "retrieval_output": retrieval_output,
            "retrieval_query": retrieval_query,
            "semantic_results": semantic_results,
            "relevant_memories": relevant_memories,
            "learning_inputs": learning_inputs,
            "learning_output": learning_output,
            "reply": reply,
            "memory_updates": memory_updates,
            "memory_reviews": memory_reviews,
            "reviewed_memories": reviewed_memories,
            "updated_working_memory": updated_working_memory,
            "working_memory_saved": working_memory_saved,
            "memory_saved": len(saved_memory_ids) > 0,
            "saved_memory_ids": saved_memory_ids,
            # Compatibility fields for previous local scripts/tests.
            "dify_inputs": learning_inputs,
            "dify_output": learning_output,
            "memory_ids": saved_memory_ids,
            "memory_update": memory_updates[0] if memory_updates else {},
            "memory_id": saved_memory_ids[0] if saved_memory_ids else None,
        }

    def _apply_memory_updates(
        self,
        user_id: str,
        memory_updates: list[dict[str, Any]],
    ) -> list[int]:
        """Save multiple Dify memory updates and index non-discard rows."""
        saved_memory_ids: list[int] = []
        for memory_update in memory_updates:
            saved_memory_id = self._apply_memory_update(user_id, memory_update)
            if saved_memory_id is not None:
                saved_memory_ids.append(saved_memory_id)
        return saved_memory_ids

    def _apply_memory_update(
        self,
        user_id: str,
        memory_update: dict[str, Any],
    ) -> int | None:
        """Save one memory update if it has content and is not discard."""
        content = str(memory_update.get("content", "")).strip()
        if not content:
            return None

        confidence = float(memory_update.get("confidence", 0.8))
        has_conflict = memory_update.get("has_conflict") is True
        memory_status = classify_memory_status(confidence, has_conflict)
        # discard -> ignored and not stored
        if memory_status == "discard":
            return None

        memory_type = memory_update.get("memory_type")
        memory_id = self.memory_store.save_memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            confidence=confidence,
            memory_status=memory_status,
        )
        self.semantic_retriever.add_memory(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            memory_status=memory_status,
            memory_type=memory_type,
        )
        return memory_id

    def _apply_memory_reviews(self, memory_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply Dify memory_reviews to SQLite and refresh retrieval metadata."""
        reviewed: list[dict[str, Any]] = []
        for memory_review in memory_reviews:
            if "memory_id" not in memory_review or "confidence" not in memory_review:
                continue

            memory_id = int(memory_review["memory_id"])
            confidence = float(memory_review["confidence"])
            has_conflict = memory_review.get("has_conflict") is True
            new_status = self.memory_store.review_memory_status(
                memory_id,
                confidence,
                has_conflict,
            )
            record = self.memory_store.get_memory_record(memory_id)
            if record and new_status == "discard":
                self.semantic_retriever.delete_memory(memory_id)
            elif record:
                self.semantic_retriever.add_memory(
                    memory_id=memory_id,
                    user_id=record["user_id"],
                    content=record["content"],
                    memory_status=record["memory_status"],
                    memory_type=record["memory_type"],
                )

            reviewed.append(
                {
                    "memory_id": memory_id,
                    "reviewed": new_status is not None,
                    "confidence": confidence,
                    "has_conflict": has_conflict,
                    "memory_status": new_status,
                }
            )
        return reviewed

    def _apply_working_memory_update(
        self,
        user_id: str,
        updated_working_memory: dict[str, Any],
    ) -> bool:
        """Store Dify's updated working memory when it provides content."""
        content = str(updated_working_memory.get("content", "")).strip()
        if not content:
            return False

        confidence = float(updated_working_memory.get("confidence", 0.8))
        self.memory_store.replace_working_memory(user_id, content, confidence)
        return True

    def review_memory(self, memory_id: int, dify_output: dict[str, Any]) -> dict[str, Any]:
        """Compatibility helper for reviewing a single memory."""
        memory_review = extract_memory_review(dify_output)
        if "confidence" not in memory_review:
            raise ValueError("memory_review.confidence is required")

        reviewed = self._apply_memory_reviews(
            [
                {
                    "memory_id": memory_id,
                    "confidence": memory_review["confidence"],
                    "has_conflict": memory_review.get("has_conflict") is True,
                }
            ]
        )
        return reviewed[0]

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        if not payload.get("user_id"):
            raise ValueError("user_id is required")
        if not isinstance(payload.get("chat_context"), dict):
            raise ValueError("chat_context is required")
