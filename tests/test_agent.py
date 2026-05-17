import tempfile
import unittest
from pathlib import Path
from typing import Any

from memory_agent.agent import MemoryAgent
from memory_agent.memory_store import MemoryStore


class FakeDifyClient:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.inputs: list[dict[str, Any]] = []

    def run_workflow(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(inputs)
        return self.output


class SequenceDifyClient:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.inputs: list[dict[str, Any]] = []

    def run_workflow(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(inputs)
        return self.outputs[len(self.inputs) - 1]


class FakeSemanticRetriever:
    def __init__(self, memory_ids: list[int] | None = None) -> None:
        self.memory_ids = memory_ids or [7]
        self.added: list[dict[str, Any]] = []
        self.deleted: list[int] = []
        self.queries: list[dict[str, Any]] = []

    def add_memory(
        self,
        memory_id: int,
        user_id: str,
        content: str,
        memory_status: str,
        memory_type: str | None,
    ) -> None:
        self.added.append(
            {
                "memory_id": memory_id,
                "user_id": user_id,
                "content": content,
                "memory_status": memory_status,
                "memory_type": memory_type,
            }
        )

    def delete_memory(self, memory_id: int) -> None:
        self.deleted.append(memory_id)

    def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.queries.append(
            {
                "user_id": user_id,
                "query_text": query_text,
                "top_k": top_k,
                "statuses": statuses,
            }
        )
        return [{"memory_id": memory_id, "score": 0.9} for memory_id in self.memory_ids]


def make_payload() -> dict[str, Any]:
    return {
        "user_id": "A001",
        "chat_context": {
            "recent_messages": [
                {"role": "target", "content": "ok"},
                {"role": "me", "content": "why so cold today"},
            ],
            "previous_recent_messages": [{"role": "target", "content": "ok"}],
        },
    }


def make_dify_output(confidence: float, has_conflict: bool = False) -> dict[str, Any]:
    return {
        "analysis": {
            "emotion": "slightly cold",
            "state": "medium willingness to reply",
        },
        "memory_update": {
            "memory_type": "conversation_style",
            "content": "user reacts weakly to frequent follow-up questions",
            "confidence": confidence,
            "has_conflict": has_conflict,
        },
    }


def make_multi_memory_dify_output() -> dict[str, Any]:
    return {
        "memory_updates": [
            {
                "memory_type": "emotion_pattern",
                "content": "gets shorter when repeatedly questioned",
                "confidence": 0.78,
                "has_conflict": False,
            },
            {
                "memory_type": "chat_style",
                "content": "usually prefers short replies",
                "confidence": 0.82,
                "has_conflict": False,
            },
            {
                "memory_type": "negative_trigger",
                "content": "dislikes frequent repeated follow-up questions",
                "confidence": 0.74,
                "has_conflict": False,
            },
        ]
    }


class MemoryAgentTest(unittest.TestCase):
    def test_process_runs_two_stage_dify_retrieval_and_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            store.replace_working_memory("A001", "current short-term state", 0.7)
            recalled_memory_id = store.save_memory(
                "A001",
                "preference",
                "known preference from sqlite",
                0.9,
                "stable",
            )
            retriever = FakeSemanticRetriever([recalled_memory_id])
            dify = SequenceDifyClient(
                [
                    {"retrieval_query": "short replies under pressure"},
                    {
                        "reply": {"should_reply": False, "content": ""},
                        "memory_updates": [
                            {
                                "memory_type": "chat_style",
                                "content": "prefers short replies",
                                "confidence": 0.82,
                                "has_conflict": False,
                            }
                        ],
                        "memory_reviews": [],
                        "updated_working_memory": {
                            "content": "updated short-term state",
                            "confidence": 0.75,
                        },
                    },
                ]
            )
            agent = MemoryAgent(store, dify, semantic_retriever=retriever)  # type: ignore[arg-type]

            result = agent.process(make_payload())

            self.assertEqual(dify.inputs[0]["stage"], "retrieval_query")
            self.assertEqual(dify.inputs[1]["stage"], "learning")
            self.assertEqual(retriever.queries[0]["query_text"], "short replies under pressure")
            self.assertEqual(
                dify.inputs[1]["info"]["relevant_memories"][0]["content"],
                "known preference from sqlite",
            )
            self.assertIn("confidence", dify.inputs[1]["info"]["relevant_memories"][0])
            self.assertEqual(result["reply"], {"should_reply": False, "content": ""})
            self.assertEqual(store.get_working_memory("A001")["content"], "updated short-term state")
            self.assertEqual(
                store.get_user_memory("A001"),
                ["known preference from sqlite", "prefers short replies"],
            )
            self.assertEqual(retriever.added[0]["memory_status"], "stable")

    def test_process_saves_stable_memory_from_dify_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            agent = MemoryAgent(store, FakeDifyClient(make_dify_output(0.82)))  # type: ignore[arg-type]

            result = agent.process(make_payload())

            self.assertEqual(result["status"], "processed")
            self.assertTrue(result["memory_saved"])
            self.assertEqual(
                store.get_user_memory("A001"),
                ["user reacts weakly to frequent follow-up questions"],
            )

    def test_low_confidence_memory_is_saved_as_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            agent = MemoryAgent(store, FakeDifyClient(make_dify_output(0.6)))  # type: ignore[arg-type]

            result = agent.process(make_payload())

            self.assertTrue(result["memory_saved"])
            self.assertEqual(store.get_user_memory("A001"), [])
            self.assertEqual(
                store.get_pending_memory("A001"),
                ["user reacts weakly to frequent follow-up questions"],
            )

    def test_conflicting_memory_is_saved_as_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            agent = MemoryAgent(
                store,
                FakeDifyClient(make_dify_output(0.95, has_conflict=True)),  # type: ignore[arg-type]
            )

            result = agent.process(make_payload())

            self.assertTrue(result["memory_saved"])
            self.assertEqual(store.get_user_memory("A001"), [])
            self.assertEqual(
                store.get_conflict_memory("A001"),
                ["user reacts weakly to frequent follow-up questions"],
            )

    def test_process_saves_multiple_memory_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            agent = MemoryAgent(store, FakeDifyClient(make_multi_memory_dify_output()))  # type: ignore[arg-type]

            result = agent.process(make_payload())

            self.assertTrue(result["memory_saved"])
            self.assertEqual(len(result["memory_ids"]), 3)
            self.assertEqual(
                store.get_user_memory("A001"),
                [
                    "gets shorter when repeatedly questioned",
                    "usually prefers short replies",
                    "dislikes frequent repeated follow-up questions",
                ],
            )

    def test_process_skips_discard_item_in_multiple_memory_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            dify_output = {
                "memory_updates": [
                    {
                        "memory_type": "stable",
                        "content": "stable memory",
                        "confidence": 0.8,
                        "has_conflict": False,
                    },
                    {
                        "memory_type": "low_quality",
                        "content": "low confidence memory",
                        "confidence": 0.3,
                        "has_conflict": False,
                    },
                ]
            }
            agent = MemoryAgent(store, FakeDifyClient(dify_output))  # type: ignore[arg-type]

            result = agent.process(make_payload())

            self.assertEqual(len(result["memory_ids"]), 1)
            self.assertEqual(store.get_user_memory("A001"), ["stable memory"])

    def test_should_save_false_is_ignored_by_new_confidence_based_save_logic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            dify_output = {
                "memory_updates": [
                    {
                        "should_save": False,
                        "memory_type": "stable",
                        "content": "saved by confidence",
                        "confidence": 0.8,
                        "has_conflict": False,
                    }
                ]
            }
            agent = MemoryAgent(store, FakeDifyClient(dify_output))  # type: ignore[arg-type]

            result = agent.process(make_payload())

            self.assertEqual(len(result["memory_ids"]), 1)
            self.assertEqual(store.get_user_memory("A001"), ["saved by confidence"])

    def test_relevant_memories_are_confirmed_by_sqlite_user_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            stable_id = store.save_memory("A001", "style", "stable sqlite memory", 0.9, "stable")
            pending_id = store.save_memory("A001", "style", "pending sqlite memory", 0.6, "pending")
            other_user_id = store.save_memory("B001", "style", "other user memory", 0.9, "stable")
            retriever = FakeSemanticRetriever([stable_id, pending_id, other_user_id])
            dify = SequenceDifyClient(
                [
                    {"retrieval_query": "sqlite memory"},
                    {
                        "reply": {},
                        "memory_updates": [],
                        "memory_reviews": [],
                        "updated_working_memory": {},
                    },
                ]
            )
            agent = MemoryAgent(store, dify, semantic_retriever=retriever)  # type: ignore[arg-type]

            result = agent.process(make_payload())

            self.assertEqual(
                result["relevant_memories"],
                [
                    {
                        "id": stable_id,
                        "user_id": "A001",
                        "memory_type": "style",
                        "content": "stable sqlite memory",
                        "confidence": 0.9,
                        "memory_status": "stable",
                        "created_at": result["relevant_memories"][0]["created_at"],
                    }
                ],
            )
            self.assertEqual(
                dify.inputs[1]["info"]["relevant_memories"],
                result["relevant_memories"],
            )

    def test_review_memory_updates_status_from_dify_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            agent = MemoryAgent(store, FakeDifyClient({}))  # type: ignore[arg-type]
            store.init_db()
            memory_id = store.save_memory(
                "A001",
                "conversation_style",
                "pending memory",
                0.6,
                "pending",
            )

            result = agent.review_memory(
                memory_id,
                {"memory_review": {"confidence": 0.82, "has_conflict": False}},
            )

            self.assertTrue(result["reviewed"])
            self.assertEqual(result["memory_status"], "stable")
            self.assertEqual(store.get_user_memory("A001"), ["pending memory"])

    def test_review_memory_can_mark_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            agent = MemoryAgent(store, FakeDifyClient({}))  # type: ignore[arg-type]
            store.init_db()
            memory_id = store.save_memory(
                "A001",
                "conversation_style",
                "stable memory",
                0.9,
                "stable",
            )

            result = agent.review_memory(
                memory_id,
                {"data": {"outputs": {"memory_review": {"confidence": 0.95, "has_conflict": True}}}},
            )

            self.assertTrue(result["reviewed"])
            self.assertEqual(result["memory_status"], "conflict")
            self.assertEqual(store.get_user_memory("A001"), [])
            self.assertEqual(store.get_conflict_memory("A001"), ["stable memory"])


if __name__ == "__main__":
    unittest.main()
