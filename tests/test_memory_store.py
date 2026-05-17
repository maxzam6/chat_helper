import tempfile
import unittest
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

            first_id = store.save_memory("A001", "style", "first fact", 0.8, "stable")
            second_id = store.save_memory("A001", "trigger", "second fact", 0.9, "stable")

            records = store.get_memory_records([second_id, 9999, first_id, second_id])

            self.assertEqual([record["id"] for record in records], [second_id, first_id])
            self.assertEqual(records[0]["memory_type"], "trigger")
            self.assertEqual(records[0]["content"], "second fact")
            self.assertEqual(records[0]["confidence"], 0.9)
            self.assertEqual(records[0]["memory_status"], "stable")
            self.assertIn("created_at", records[0])

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


if __name__ == "__main__":
    unittest.main()
