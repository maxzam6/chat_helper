import time
import unittest
from typing import Any

from memory_agent.hotkey_capture import DISPLAY_HOTKEY, HotkeyCaptureService


class FakeAgent:
    def __init__(self, calls: list[dict[str, Any]], delay: float = 0.0) -> None:
        self.calls = calls
        self.delay = delay

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(payload))
        if self.delay:
            time.sleep(self.delay)
        return {
            "status": "processed",
            "reply": {"content": payload.get("user_input", "")},
        }


def wait_for(predicate, timeout: float = 2.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = predicate()
        if last.get("done"):
            return last
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for condition. Last state: {last}")


class HotkeyCaptureServiceTest(unittest.TestCase):
    def test_empty_user_input_returns_prompt_without_running_agent(self):
        calls: list[dict[str, Any]] = []
        service = HotkeyCaptureService(lambda: FakeAgent(calls))

        service.update_context({"user_input": "", "screenshot_region": {"left": 1}})
        service.trigger()

        state = wait_for(
            lambda: {
                "done": service.status()["latest_result_generation"] == 1,
                "status": service.status(),
            }
        )["status"]

        self.assertEqual(state["hotkey"], DISPLAY_HOTKEY)
        self.assertEqual(state["latest_result"]["status"], "missing_user_input")
        self.assertEqual(calls, [])

    def test_cancelled_generation_does_not_block_next_trigger(self):
        calls: list[dict[str, Any]] = []
        service = HotkeyCaptureService(lambda: FakeAgent(calls, delay=0.15))

        service.update_context({"user_input": "first"})
        service.trigger()
        service.cancel()
        service.update_context({"user_input": "second"})
        service.trigger()

        state = wait_for(
            lambda: {
                "done": service.status()["latest_result_generation"] == 2,
                "status": service.status(),
            }
        )["status"]

        self.assertFalse(state["running"])
        self.assertFalse(state["cancel_requested"])
        self.assertEqual(state["latest_result"]["reply"]["content"], "second")
        self.assertEqual(state["latest_result_generation"], 2)


if __name__ == "__main__":
    unittest.main()
