import json
import unittest

from memory_agent.models import (
    extract_memory_review,
    extract_memory_reviews,
    extract_memory_update,
    extract_memory_updates,
    extract_reply,
    extract_retrieval_query,
    extract_updated_working_memory,
    parse_llm_json,
)


class ModelsTest(unittest.TestCase):
    def test_parse_llm_json_from_result_string(self):
        payload = {"retrieval_query": "relationship pressure"}

        self.assertEqual(parse_llm_json(json.dumps(payload)), payload)

    def test_parse_llm_json_from_dict(self):
        payload = {"reply": {"should_reply": True, "content": "ok"}}

        self.assertEqual(parse_llm_json(payload), payload)

    def test_parse_llm_json_from_markdown_json_block(self):
        result = """```json
{
  "retrieval_query": "negative trigger"
}
```"""

        self.assertEqual(parse_llm_json(result), {"retrieval_query": "negative trigger"})

    def test_parse_llm_json_returns_empty_on_bad_content(self):
        self.assertEqual(parse_llm_json(None), {})
        self.assertEqual(parse_llm_json("not json"), {})
        self.assertEqual(parse_llm_json("[1, 2, 3]"), {})

    def test_extract_retrieval_query_supports_direct_outputs_and_result(self):
        self.assertEqual(extract_retrieval_query({"retrieval_query": "direct"}), "direct")
        self.assertEqual(
            extract_retrieval_query({"data": {"outputs": {"retrieval_query": "outputs"}}}),
            "outputs",
        )
        self.assertEqual(
            extract_retrieval_query({"result": '{"retrieval_query": "result"}'}),
            "result",
        )

    def test_extract_learning_fields_from_result_string(self):
        result = {
            "reply": {
                "should_reply": True,
                "content": "hello",
                "reason": "enough context",
            },
            "memory_updates": [
                {
                    "memory_type": "emotion_pattern",
                    "content": "shorter replies under pressure",
                    "confidence": 0.78,
                    "has_conflict": False,
                }
            ],
            "memory_reviews": [
                {
                    "memory_id": 12,
                    "confidence": 0.88,
                    "has_conflict": False,
                }
            ],
            "updated_working_memory": {
                "content": "current relationship state",
                "confidence": 0.74,
            },
        }
        output = {"result": json.dumps(result)}

        self.assertEqual(extract_reply(output), result["reply"])
        self.assertEqual(extract_memory_updates(output), result["memory_updates"])
        self.assertEqual(extract_memory_reviews(output), result["memory_reviews"])
        self.assertEqual(extract_updated_working_memory(output), result["updated_working_memory"])

    def test_extract_legacy_single_fields_from_result_string(self):
        result = {
            "memory_update": {
                "memory_type": "chat_style",
                "content": "prefers short replies",
                "confidence": 0.82,
            },
            "memory_review": {
                "memory_id": 3,
                "confidence": 0.9,
            },
        }
        output = {"result": json.dumps(result)}

        self.assertEqual(extract_memory_update(output), result["memory_update"])
        self.assertEqual(extract_memory_review(output), result["memory_review"])
        self.assertEqual(extract_memory_updates(output), [result["memory_update"]])
        self.assertEqual(extract_memory_reviews(output), [result["memory_review"]])

    def test_old_formats_keep_priority_over_result_json(self):
        output = {
            "reply": {"content": "direct"},
            "result": '{"reply": {"content": "result"}}',
        }

        self.assertEqual(extract_reply(output), {"content": "direct"})


if __name__ == "__main__":
    unittest.main()
