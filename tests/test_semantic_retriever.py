import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_agent.semantic_retriever import SemanticRetriever


class SemanticRetrieverTest(unittest.TestCase):
    def test_fallback_returns_only_overlapping_memory_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(SemanticRetriever, "_init_chroma", return_value=None):
                retriever = SemanticRetriever(persist_path=Path(tmp) / "chroma")
            retriever._available = False

            retriever.add_memory(1, "A001", "short replies under pressure", "stable", "style")
            retriever.add_memory(2, "A001", "likes weekend hiking", "stable", "preference")
            retriever.add_memory(3, "A001", "short replies but pending", "pending", "style")

            results = retriever.query(
                user_id="A001",
                query_text="short replies",
                statuses=["stable"],
            )

            self.assertEqual(results, [{"memory_id": 1, "score": 2.0}])

    def test_fallback_returns_empty_when_no_token_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(SemanticRetriever, "_init_chroma", return_value=None):
                retriever = SemanticRetriever(persist_path=Path(tmp) / "chroma")
            retriever._available = False

            retriever.add_memory(1, "A001", "likes weekend hiking", "stable", "preference")

            self.assertEqual(
                retriever.query("A001", "short replies", statuses=["stable"]),
                [],
            )


if __name__ == "__main__":
    unittest.main()
