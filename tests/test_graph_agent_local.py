import tempfile
import unittest
from pathlib import Path
from typing import Any

from memory_agent.graph_agent import GraphMemoryAgent
from memory_agent.llm_client import MockLLMClient
from memory_agent.memory_store import MemoryStore


class RecordingLLMClient(MockLLMClient):
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def generate_json(self, task: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        return super().generate_json(task, inputs)


class EmptyRetriever:
    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []
        self.deleted: list[int] = []

    def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []

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

    def embed_text(self, text: str) -> list[float] | None:
        return None

    def cosine_similarity(
        self,
        vec1: list[float] | None,
        vec2: list[float] | None,
    ) -> float:
        return 0.0


def make_chat_context() -> dict[str, Any]:
    return {
        "recent_messages": [
            {"role": "user", "content": "哦"},
            {"role": "me", "content": "你是不是不想聊了"},
        ],
        "previous_recent_messages": [],
    }


class GraphMemoryAgentLocalTest(unittest.TestCase):
    def test_general_question_does_not_require_current_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent = GraphMemoryAgent(store, llm, semantic_retriever=EmptyRetriever())  # type: ignore[arg-type]

            result = agent.process({"me_id": "default", "user_input": "LangGraph 是什么？"})

            self.assertEqual(result["intent"], "general_question")
            self.assertTrue(result["reply"])
            self.assertEqual(llm.tasks, ["intent_classifier", "reply"])
            self.assertIsNotNone(store.get_session_state("default", "global"))
            self.assertEqual(store.get_user_memory("A001"), [])

    def test_revise_reply_uses_last_session_reply_without_retrieval_or_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent = GraphMemoryAgent(store, llm, semantic_retriever=EmptyRetriever())  # type: ignore[arg-type]

            agent.process({"me_id": "default", "user_input": "LangGraph 是什么？"})
            llm.tasks.clear()
            result = agent.process({"me_id": "default", "user_input": "短一点，自然一点"})

            self.assertEqual(result["intent"], "revise_reply")
            self.assertTrue(result["reply"])
            self.assertEqual(llm.tasks, ["intent_classifier", "reply"])
            self.assertNotIn("ocr", llm.tasks)
            self.assertNotIn("retrieval_query", llm.tasks)
            self.assertNotIn("learning", llm.tasks)

    def test_reply_advice_runs_memory_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            retriever = EmptyRetriever()
            agent = GraphMemoryAgent(store, llm, semantic_retriever=retriever)  # type: ignore[arg-type]

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": "她回我哦，我该怎么回？",
                    "chat_context": make_chat_context(),
                    "working_memory_observations": [
                        {"content": "The current chat needs a soft reply.", "confidence": 0.8}
                    ],
                }
            )

            self.assertEqual(result["intent"], "reply_advice")
            self.assertIn("retrieval_query", llm.tasks)
            self.assertIn("learning", llm.tasks)
            self.assertTrue(result["reply"])
            self.assertTrue(result["saved_memory_ids"])
            self.assertTrue(store.get_user_memory("A001"))
            self.assertTrue(store.get_working_memory_observations("A001"))
            self.assertIsNotNone(store.get_session_state("default", "A001"))
            self.assertTrue(retriever.added)

    def test_profile_update_skips_ocr_and_updates_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent = GraphMemoryAgent(store, llm, semantic_retriever=EmptyRetriever())  # type: ignore[arg-type]

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": "记一下，她平时话少，不是冷淡",
                }
            )

            self.assertEqual(result["intent"], "profile_update")
            self.assertNotIn("ocr", llm.tasks)
            self.assertIn("retrieval_query", llm.tasks)
            self.assertIn("learning", llm.tasks)
            self.assertTrue(result["saved_memory_ids"])
            self.assertTrue(result["reply"])
            self.assertTrue(store.get_user_memory("A001"))


if __name__ == "__main__":
    unittest.main()
