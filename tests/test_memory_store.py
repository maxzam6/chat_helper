import tempfile
import unittest
import sqlite3
from pathlib import Path

from memory_agent.memory_store import MemoryStore, classify_memory_status


class MemoryStoreTest(unittest.TestCase):
    def test_classify_memory_status(self):
        self.assertEqual(classify_memory_status(0.9), "stable")
        self.assertEqual(classify_memory_status(0.7), "stable")
        self.assertEqual(classify_memory_status(0.6), "pending")
        self.assertEqual(classify_memory_status(0.5), "pending")
        self.assertEqual(classify_memory_status(0.49), "discard")
        self.assertEqual(classify_memory_status(0.9, has_conflict=True), "conflict")

    def test_save_and_get_memory_returns_only_stable_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            stable_id = store.save_memory(
                "A001",
                "conversation_style",
                "likes short replies",
                0.9,
                "stable",
            )
            store.save_memory("A001", "conversation_style", "needs review", 0.6, "pending")
            store.save_memory("A001", "conversation_style", "conflicting fact", 0.8, "conflict")
            store.save_memory("A001", "conversation_style", "low quality fact", 0.2, "discard")

            self.assertGreater(stable_id, 0)
            self.assertEqual(store.get_user_memory("A001"), ["likes short replies"])
            self.assertEqual(store.get_user_memory("B001"), [])

    def test_get_pending_and_conflict_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            store.save_memory("A001", "conversation_style", "pending fact", 0.6, "pending")
            store.save_memory("A001", "conversation_style", "conflict fact", 0.8, "conflict")

            self.assertEqual(store.get_pending_memory("A001"), ["pending fact"])
            self.assertEqual(store.get_conflict_memory("A001"), ["conflict fact"])

    def test_get_memory_records_returns_full_records_in_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            first_id = store.save_memory(
                "A001",
                "style",
                "first fact",
                0.8,
                "stable",
                source_type="reply_advice",
                source_summary="first summary",
                last_evidence="first evidence",
            )
            second_id = store.save_memory("A001", "trigger", "second fact", 0.9, "stable")

            records = store.get_memory_records([second_id, 9999, first_id, second_id])

            self.assertEqual([record["id"] for record in records], [second_id, first_id])
            self.assertEqual(records[0]["memory_type"], "trigger")
            self.assertEqual(records[0]["content"], "second fact")
            self.assertEqual(records[0]["confidence"], 0.9)
            self.assertEqual(records[0]["memory_status"], "stable")
            self.assertIn("created_at", records[0])
            self.assertIn("updated_at", records[0])
            self.assertEqual(records[1]["source_type"], "reply_advice")
            self.assertEqual(records[1]["source_summary"], "first summary")
            self.assertEqual(records[1]["last_evidence"], "first evidence")

    def test_update_memory_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            memory_id = store.save_memory("A001", "conversation_style", "pending fact", 0.6, "pending")

            self.assertEqual(store.get_user_memory("A001"), [])
            self.assertTrue(store.update_memory_status(memory_id, "stable"))
            self.assertEqual(store.get_user_memory("A001"), ["pending fact"])
            self.assertFalse(store.update_memory_status(9999, "stable"))

    def test_review_memory_status_updates_confidence_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            memory_id = store.save_memory("A001", "conversation_style", "pending fact", 0.6, "pending")

            self.assertEqual(store.review_memory_status(memory_id, 0.8), "stable")
            self.assertEqual(store.get_user_memory("A001"), ["pending fact"])
            self.assertEqual(store.review_memory_status(memory_id, 0.9, has_conflict=True), "conflict")
            self.assertEqual(store.get_conflict_memory("A001"), ["pending fact"])
            self.assertIsNone(store.review_memory_status(9999, 0.8))

    def test_update_memory_review_updates_confidence_status_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            memory_id = store.save_memory("A001", "style", "review fact", 0.8, "stable")
            self.assertTrue(store.update_memory_review(memory_id, 0.4, "discard"))
            record = store.get_memory_record(memory_id)

            self.assertEqual(record["confidence"], 0.4)
            self.assertEqual(record["memory_status"], "discard")
            self.assertIsNotNone(record["updated_at"])

    def test_init_db_migrates_old_user_memory_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE user_memory (
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
                    INSERT INTO user_memory (
                        user_id,
                        memory_type,
                        content,
                        confidence,
                        memory_status
                    )
                    VALUES ('A001', 'style', 'old fact', 0.8, 'stable')
                    """
                )
                conn.commit()
            finally:
                conn.close()

            store = MemoryStore(db_path)
            store.init_db()
            old_record = store.get_memory_record(1)
            new_id = store.save_memory("A001", "style", "new fact", 0.9, "stable")
            new_record = store.get_memory_record(new_id)

            self.assertIn("source_type", old_record)
            self.assertIsNotNone(old_record["updated_at"])
            self.assertIsNotNone(new_record["updated_at"])

    def test_invalid_memory_status_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            with self.assertRaises(ValueError):
                store.save_memory("A001", "conversation_style", "bad status", 0.8, "unknown")

            with self.assertRaises(ValueError):
                store.update_memory_status(1, "unknown")

    def test_replace_and_get_working_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            self.assertIsNone(store.get_working_memory("A001"))

            store.replace_working_memory("A001", "short-term conversation state", 0.74)
            first = store.get_working_memory("A001")
            self.assertEqual(first["content"], "short-term conversation state")
            self.assertEqual(first["confidence"], 0.74)

            store.replace_working_memory("A001", "updated state", 0.8)
            second = store.get_working_memory("A001")
            self.assertEqual(second["content"], "updated state")
            self.assertEqual(second["confidence"], 0.8)

    def test_working_memory_observations_age_cleanup_and_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            first = store.update_working_memory_observations(
                "A001",
                [
                    {"content": "first", "confidence": 0.5, "ttl": 2},
                    {"content": "second", "confidence": 0.9, "ttl": 5},
                ],
                max_items=2,
            )

            self.assertEqual([item["content"] for item in first], ["second", "first"])

            second = store.update_working_memory_observations(
                "A001",
                [{"content": "third", "confidence": 0.8, "ttl": 5}],
                max_items=2,
            )

            self.assertEqual([item["content"] for item in second], ["second", "third"])

    def test_session_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init_db()

            store.save_session_state(
                "me",
                "A001",
                {
                    "last_intent": "reply_advice",
                    "last_user_input": "input",
                    "last_input_summary": "summary",
                    "last_reply": {"content": "reply"},
                    "last_analysis": {"state": "ok"},
                    "last_chat_context": {"recent_messages": []},
                    "last_active_user_id": "A001",
                    "last_retrieval_query": "query",
                },
            )

            session = store.get_session_state("me", "A001")

            self.assertEqual(session["last_reply"], {"content": "reply"})
            self.assertEqual(session["last_analysis"], {"state": "ok"})
            self.assertEqual(session["last_chat_context"], {"recent_messages": []})
            self.assertEqual(session["last_retrieval_query"], "query")


if __name__ == "__main__":
    unittest.main()
