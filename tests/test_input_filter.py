import unittest

from memory_agent.input_filter import InputFilter


class InputFilterTest(unittest.TestCase):
    def test_empty_recent_messages(self):
        result = InputFilter().check(
            {
                "chat_context": {
                    "recent_messages": [],
                    "previous_recent_messages": [],
                }
            }
        )

        self.assertFalse(result.should_process)
        self.assertEqual(result.reason, "empty_recent_messages")

    def test_duplicate_recent_messages(self):
        messages = [{"role": "target", "content": "哦"}]

        result = InputFilter().check(
            {
                "chat_context": {
                    "recent_messages": messages,
                    "previous_recent_messages": messages,
                }
            }
        )

        self.assertFalse(result.should_process)
        self.assertEqual(result.reason, "duplicate_recent_messages")

    def test_new_recent_messages(self):
        result = InputFilter().check(
            {
                "chat_context": {
                    "recent_messages": [{"role": "target", "content": "没有啊"}],
                    "previous_recent_messages": [{"role": "target", "content": "哦"}],
                }
            }
        )

        self.assertTrue(result.should_process)


if __name__ == "__main__":
    unittest.main()

