from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from memory_agent.active_memory_cache import ActiveMemoryCache
from memory_agent.graph_agent import GraphMemoryAgent
from memory_agent.input_filter import InputFilter
from memory_agent.llm_client import MockLLMClient
from memory_agent.memory_store import MemoryStore
from memory_agent.semantic_retriever import SemanticRetriever


class RecordingLLMClient(MockLLMClient):
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def generate_json(self, task: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.tasks.append(task)
        return super().generate_json(task, inputs)


class RecordingSemanticRetriever(SemanticRetriever):
    """SemanticRetriever test double that avoids optional Chroma/model setup."""

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
        statuses = statuses or ["stable"]
        deleted_ids = set(self.deleted)
        results: list[dict[str, Any]] = []
        for item in self.added:
            if item["memory_id"] in deleted_ids:
                continue
            if item["user_id"] != user_id:
                continue
            if item["memory_status"] not in statuses:
                continue
            results.append({"memory_id": item["memory_id"], "score": 1.0})
        return results[:top_k]

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


class FailingAddRetriever(RecordingSemanticRetriever):
    def add_memory(
        self,
        memory_id: int,
        user_id: str,
        content: str,
        memory_status: str,
        memory_type: str | None,
    ) -> None:
        raise RuntimeError("index unavailable")


def make_agent(
    store: MemoryStore,
    llm: RecordingLLMClient,
    retriever: RecordingSemanticRetriever | None = None,
) -> tuple[GraphMemoryAgent, RecordingSemanticRetriever]:
    retriever = retriever or RecordingSemanticRetriever()
    agent = GraphMemoryAgent(
        memory_store=store,
        llm_client=llm,
        input_filter=InputFilter(),
        semantic_retriever=retriever,
        active_memory_cache=ActiveMemoryCache(),
    )
    return agent, retriever


def make_chat_context() -> dict[str, Any]:
    return {
        "recent_messages": [
            {"role": "user", "content": "\u54e6"},
            {"role": "me", "content": "\u4f60\u662f\u4e0d\u662f\u4e0d\u60f3\u804a\u4e86"},
        ],
        "previous_recent_messages": [],
    }


class GraphMemoryAgentLocalTest(unittest.TestCase):
    def test_process_returns_public_result_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)

            result = agent.process(
                {
                    "me_id": "default",
                    "user_input": "LangGraph \u662f\u4ec0\u4e48\uff1f",
                }
            )

            self.assertEqual(result["intent"], "general_question")
            self.assertEqual(result["task_list"], ["general_question"])
            self.assertEqual(result["completed_tasks"], ["general_question"])
            self.assertEqual([item["task"] for item in result["task_results"]], ["general_question"])
            self.assertIn("debug", result)
            self.assertNotIn("memory_updates", result)
            self.assertTrue(result["session_state_saved"])

    def test_general_question_does_not_require_current_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)

            result = agent.process(
                {
                    "me_id": "default",
                    "user_input": "LangGraph \u662f\u4ec0\u4e48\uff1f",
                },
                return_full_state=True,
            )

            self.assertEqual(result["intent"], "general_question")
            self.assertTrue(result["reply"])
            self.assertTrue(result["session_state_saved"])
            self.assertEqual(llm.tasks, ["intent_classifier", "reply"])
            self.assertIsNotNone(store.get_session_state("default", "global"))
            self.assertEqual(store.get_user_memory("A001"), [])

    def test_revise_reply_uses_last_session_reply_without_retrieval_or_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)

            agent.process(
                {"me_id": "default", "user_input": "LangGraph \u662f\u4ec0\u4e48\uff1f"},
                return_full_state=True,
            )
            llm.tasks.clear()
            result = agent.process(
                {
                    "me_id": "default",
                    "user_input": "\u77ed\u4e00\u70b9\uff0c\u81ea\u7136\u4e00\u70b9",
                },
                return_full_state=True,
            )

            self.assertEqual(result["intent"], "revise_reply")
            self.assertTrue(result["reply"])
            self.assertEqual(llm.tasks, ["intent_classifier", "reply"])
            self.assertNotIn("ocr", llm.tasks)
            self.assertNotIn("retrieval_query", llm.tasks)
            self.assertNotIn("learning", llm.tasks)

    def test_reply_advice_runs_memory_flow_with_existing_chat_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, retriever = make_agent(store, llm)

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": "\u5979\u56de\u6211\u54e6\uff0c\u6211\u8be5\u600e\u4e48\u56de\uff1f",
                    "chat_context": make_chat_context(),
                },
                return_full_state=True,
            )

            self.assertEqual(result["intent"], "reply_advice")
            self.assertIn("ocr", llm.tasks)
            self.assertIn("retrieval_query", llm.tasks)
            self.assertIn("learning", llm.tasks)
            self.assertTrue(result["working_memory"])
            self.assertTrue(result["retrieval_query"])
            self.assertTrue(result["reply"])
            self.assertTrue(result["memory_updates"])
            self.assertIsInstance(result["saved_memory_ids"], list)
            self.assertFalse(any(str(memory["id"]).startswith("tmp_") for memory in result["active_memory_cache"]["memories"]))
            self.assertEqual(result["active_memory_cache"]["dirty_memory_ids"], [])
            self.assertTrue(result["session_state_saved"])
            self.assertTrue(store.get_user_memory("A001"))
            self.assertTrue(store.get_working_memory_observations("A001"))
            self.assertIsNotNone(store.get_session_state("default", "A001"))
            self.assertTrue(retriever.added)

    def test_profile_update_skips_ocr_and_updates_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": "\u8bb0\u4e00\u4e0b\uff0c\u5979\u5e73\u65f6\u8bdd\u5c11\uff0c\u4e0d\u662f\u51b7\u6de1",
                },
                return_full_state=True,
            )

            self.assertEqual(result["intent"], "profile_update")
            self.assertNotIn("ocr", llm.tasks)
            self.assertIn("retrieval_query", llm.tasks)
            self.assertIn("learning", llm.tasks)
            self.assertTrue(result["retrieval_query"])
            self.assertTrue(result["memory_updates"])
            self.assertTrue(result["saved_memory_ids"])
            self.assertTrue(result["reply"])
            self.assertTrue(result["session_state_saved"])
            self.assertTrue(store.get_user_memory("A001"))

    def test_unknown_similar_user_id_returns_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            store.save_memory("A001", "style", "known user memory", 0.8, "stable")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A01",
                    "user_input": "\u5979\u56de\u6211\u54e6\uff0c\u6211\u8be5\u600e\u4e48\u56de\uff1f",
                    "chat_context": make_chat_context(),
                },
                return_full_state=True,
            )

            self.assertEqual(result["active_user_id"], "A01")
            self.assertTrue(result["user_id_suggestions"])
            self.assertEqual(result["user_id_suggestions"][0]["user_id"], "A001")

    def test_skipped_input_does_not_overwrite_last_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)

            first = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": "\u5979\u56de\u6211\u54e6\uff0c\u6211\u8be5\u600e\u4e48\u56de\uff1f",
                    "chat_context": make_chat_context(),
                },
                return_full_state=True,
            )
            self.assertTrue(first["session_state_saved"])
            original_reply = store.get_session_state("default", "A001")["last_reply"]

            skipped = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": "\u5979\u56de\u6211\u54e6\uff0c\u6211\u8be5\u600e\u4e48\u56de\uff1f",
                    "chat_context": {
                        "recent_messages": [],
                        "previous_recent_messages": [],
                    },
                },
                return_full_state=True,
            )

            self.assertEqual(skipped["status"], "skipped")
            self.assertFalse(skipped["session_state_saved"])
            self.assertEqual(store.get_session_state("default", "A001")["last_reply"], original_reply)

            llm.tasks.clear()
            revised = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": "\u77ed\u4e00\u70b9",
                },
                return_full_state=True,
            )

            self.assertEqual(revised["intent"], "revise_reply")
            self.assertTrue(revised["reply"])
            self.assertEqual(llm.tasks, ["intent_classifier", "reply"])

    def test_memory_update_enters_cache_before_sync_writeback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            llm = RecordingLLMClient()
            agent, retriever = make_agent(store, llm)
            agent.active_memory_cache.set_cache("A001", "short replies", [])

            state = {
                "active_user_id": "A001",
                "current_user_id": "A001",
                "memory_updates": [
                    {
                        "memory_type": "conversation_style",
                        "content": "prefers concise replies",
                        "confidence": 0.82,
                        "has_conflict": False,
                    }
                ],
                "memory_reviews": [],
            }

            cache_update = agent.update_cache_from_learning(state)

            self.assertEqual(cache_update["saved_memory_ids"], [])
            self.assertEqual(store.get_user_memory("A001"), [])
            self.assertTrue(cache_update["dirty_memories"])
            temp_memory = cache_update["dirty_memories"][0]
            self.assertTrue(temp_memory["_is_new"])
            self.assertTrue(str(temp_memory["id"]).startswith("tmp_"))

            sync_result = agent.sync_dirty_memory({})
            saved_id = sync_result["saved_memory_ids"][0]

            self.assertEqual(store.get_user_memory("A001"), ["prefers concise replies"])
            self.assertEqual(retriever.query("A001", "concise", statuses=["stable"])[0]["memory_id"], saved_id)
            cache_ids = [memory["id"] for memory in agent.active_memory_cache.memories]
            self.assertIn(saved_id, cache_ids)
            self.assertFalse(any(str(memory_id).startswith("tmp_") for memory_id in cache_ids))
            self.assertEqual(agent.active_memory_cache.dirty_memory_ids, set())

    def test_duplicate_memory_update_reuses_existing_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            existing_id = store.save_memory(
                "A001",
                "conversation_style",
                "prefers concise replies",
                0.82,
                "stable",
            )
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)
            agent.active_memory_cache.set_cache("A001", "short replies", [])
            agent.update_cache_from_learning(
                {
                    "active_user_id": "A001",
                    "current_user_id": "A001",
                    "memory_updates": [
                        {
                            "memory_type": "conversation_style",
                            "content": "prefers concise replies",
                            "confidence": 0.82,
                            "has_conflict": False,
                        }
                    ],
                    "memory_reviews": [],
                }
            )

            sync_result = agent.sync_dirty_memory({})

            self.assertEqual(sync_result["saved_memory_ids"], [])
            self.assertEqual(store.get_user_memory("A001"), ["prefers concise replies"])
            self.assertEqual(
                len(store.get_memory_records([existing_id])),
                1,
            )
            self.assertEqual(agent.active_memory_cache.get_memories()[0]["id"], existing_id)
            self.assertTrue(sync_result["reviewed_memories"][0]["duplicate"])

    def test_memory_review_updates_cache_then_syncs_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            memory_id = store.save_memory(
                "A001",
                "belief",
                "\u7528\u6237\u5bb9\u6613\u628a\u77ed\u56de\u590d\u8bef\u5224\u4e3a\u51b7\u6de1",
                0.8,
                "stable",
            )
            llm = RecordingLLMClient()
            agent, retriever = make_agent(store, llm)
            record = store.get_memory_record(memory_id)
            retriever.add_memory(memory_id, "A001", record["content"], "stable", "belief")
            agent.active_memory_cache.set_cache("A001", "short reply", [record])

            cache_update = agent.update_cache_from_learning(
                {
                    "memory_updates": [],
                    "memory_reviews": [
                        {
                            "memory_id": memory_id,
                            "confidence": 0.4,
                            "has_conflict": True,
                        }
                    ],
                }
            )

            cached = agent.active_memory_cache.get_memories()[0]
            self.assertEqual(cached["memory_status"], "conflict")
            self.assertEqual(cached["confidence"], 0.4)
            self.assertTrue(cache_update["dirty_memories"])
            self.assertEqual(store.get_memory_record(memory_id)["memory_status"], "stable")

            agent.sync_dirty_memory({})

            self.assertEqual(store.get_memory_record(memory_id)["memory_status"], "conflict")
            self.assertEqual(agent.active_memory_cache.get_memories()[0]["memory_status"], "conflict")
            self.assertEqual(agent.active_memory_cache.dirty_memory_ids, set())

    def test_discard_review_removes_memory_from_cache_and_retriever(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            memory_id = store.save_memory(
                "A001",
                "belief",
                "low quality belief",
                0.8,
                "stable",
            )
            llm = RecordingLLMClient()
            agent, retriever = make_agent(store, llm)
            record = store.get_memory_record(memory_id)
            retriever.add_memory(memory_id, "A001", record["content"], "stable", "belief")
            agent.active_memory_cache.set_cache("A001", "low quality", [record])

            agent.update_cache_from_learning(
                {
                    "memory_updates": [],
                    "memory_reviews": [
                        {
                            "memory_id": memory_id,
                            "confidence": 0.4,
                            "has_conflict": False,
                        }
                    ],
                }
            )
            agent.sync_dirty_memory({})

            self.assertEqual(store.get_memory_record(memory_id)["memory_status"], "discard")
            self.assertIn(memory_id, retriever.deleted)
            self.assertEqual(agent.active_memory_cache.get_memories(["stable"]), [])
            self.assertEqual(agent.active_memory_cache.dirty_memory_ids, set())

    def test_user_switch_syncs_dirty_cache_before_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)
            agent.active_memory_cache.set_cache("A001", "old user", [])
            agent.active_memory_cache.add_pending_memory(
                {
                    "user_id": "A001",
                    "memory_type": "conversation_style",
                    "content": "A001 pending memory",
                    "confidence": 0.8,
                    "memory_status": "stable",
                    "source_type": "reply_advice",
                    "source_summary": "old summary",
                }
            )

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A002",
                    "user_input": "\u8bb0\u4e00\u4e0b\uff0c\u5979\u5e73\u65f6\u8bdd\u5c11\uff0c\u4e0d\u662f\u51b7\u6de1",
                },
                return_full_state=True,
            )

            self.assertEqual(result["active_user_id"], "A002")
            self.assertTrue(result["context_switch_sync_result"])
            self.assertTrue(result["context_switch_sync_result"]["saved_memory_ids"])
            self.assertEqual(store.get_user_memory("A001"), ["A001 pending memory"])
            self.assertEqual(agent.active_memory_cache.user_id, "A002")

    def test_user_switch_continues_when_dirty_sync_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm, FailingAddRetriever())
            agent.active_memory_cache.set_cache("A001", "old user", [])
            agent.active_memory_cache.add_pending_memory(
                {
                    "user_id": "A001",
                    "memory_type": "conversation_style",
                    "content": "A001 pending memory",
                    "confidence": 0.8,
                    "memory_status": "stable",
                    "source_type": "reply_advice",
                    "source_summary": "old summary",
                }
            )

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A002",
                    "user_input": "\u8bb0\u4e00\u4e0b\uff0c\u5979\u5e73\u65f6\u8bdd\u5c11\uff0c\u4e0d\u662f\u51b7\u6de1",
                },
                return_full_state=True,
            )

            self.assertEqual(result["active_user_id"], "A002")
            self.assertEqual(agent.active_memory_cache.user_id, "A002")
            self.assertTrue(result["context_switch_sync_result"]["sync_errors"])
            self.assertTrue(result["context_switch_sync_result"]["forced_user_switch"])
            self.assertIn("context_switch_sync_failed", result["error"])
            self.assertEqual(store.get_user_memory("A001"), ["A001 pending memory"])

    def test_sync_error_keeps_dirty_memory_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm, FailingAddRetriever())
            agent.active_memory_cache.set_cache("A001", "short replies", [])
            agent.update_cache_from_learning(
                {
                    "active_user_id": "A001",
                    "current_user_id": "A001",
                    "intent": "reply_advice",
                    "input_summary": "summary",
                    "memory_updates": [
                        {
                            "memory_type": "conversation_style",
                            "content": "needs retry",
                            "confidence": 0.82,
                            "has_conflict": False,
                        }
                    ],
                    "memory_reviews": [],
                }
            )

            sync_result = agent.sync_dirty_memory({})

            self.assertTrue(sync_result["sync_errors"])
            self.assertTrue(agent.active_memory_cache.dirty_memory_ids)
            self.assertEqual(store.get_user_memory("A001"), ["needs retry"])

    def test_multiple_intents_run_in_priority_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            llm = RecordingLLMClient()
            agent, _ = make_agent(store, llm)

            result = agent.process(
                {
                    "me_id": "default",
                    "current_user_id": "A001",
                    "user_input": (
                        "LangGraph \u662f\u4ec0\u4e48\uff1f"
                        "\u5979\u56de\u6211\u54e6\uff0c\u6211\u8be5\u600e\u4e48\u56de\uff1f"
                        "\u8bb0\u4e00\u4e0b\uff0c\u5979\u5e73\u65f6\u8bdd\u5c11\uff0c\u4e0d\u662f\u51b7\u6de1"
                    ),
                    "chat_context": make_chat_context(),
                },
                return_full_state=True,
            )

            expected_tasks = ["general_question", "reply_advice", "profile_update"]
            self.assertEqual(result["task_list"], expected_tasks)
            self.assertEqual(result["completed_tasks"], expected_tasks)
            self.assertEqual([item["task"] for item in result["task_results"]], expected_tasks)
            self.assertEqual(result["intent"], "profile_update")
            self.assertEqual(llm.tasks.count("intent_classifier"), 1)
            self.assertEqual(llm.tasks.count("ocr"), 1)
            self.assertEqual(llm.tasks.count("retrieval_query"), 2)
            self.assertEqual(llm.tasks.count("learning"), 2)
            self.assertGreaterEqual(llm.tasks.count("reply"), 3)
            self.assertTrue(result["saved_memory_ids"])
            self.assertGreaterEqual(len(store.get_user_memory("A001")), 2)


if __name__ == "__main__":
    unittest.main()
