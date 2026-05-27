from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .graph_agent import GraphMemoryAgent
from tools.screenshot_tool import capture_region_as_base64


DEFAULT_HOTKEY = "<ctrl>+<shift>+y"
DISPLAY_HOTKEY = "Ctrl + Shift + Y"


class HotkeyCaptureService:
    """Global hotkey service for fixed-region chat capture.

    The browser no longer needs to be focused when analysis starts. The frontend
    only updates the pending payload; pressing the hotkey triggers the backend
    capture and GraphMemoryAgent flow.
    """

    def __init__(
        self,
        agent_factory: Callable[[], GraphMemoryAgent],
        hotkey: str = DEFAULT_HOTKEY,
    ) -> None:
        self.agent_factory = agent_factory
        self.hotkey = hotkey
        self.display_hotkey = DISPLAY_HOTKEY
        self.context: dict[str, Any] = {}
        self.latest_result: dict[str, Any] | None = None
        self.latest_progress: dict[str, Any] | None = None
        self.latest_error: str | None = None
        self.running = False
        self.cancel_requested = False
        self.trigger_count = 0
        self.latest_result_generation = 0
        self.latest_progress_generation = 0
        self.updated_at: float | None = None
        self._lock = threading.Lock()
        self._listener: Any = None

    def start(self) -> None:
        """Start global hotkey listening. Missing dependency is reported in status."""
        if self._listener is not None:
            return
        try:
            from pynput import keyboard
        except Exception as exc:
            self.latest_error = f"hotkey_dependency_missing:{exc}"
            return
        self._listener = keyboard.GlobalHotKeys({self.hotkey: self.trigger})
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def update_context(self, context: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.context = dict(context)
            self.latest_error = None
            self.cancel_requested = False
            self.updated_at = time.time()
        return self.status()

    def trigger(self) -> dict[str, Any]:
        with self._lock:
            if self.running:
                return self.status()
            self.running = True
            self.cancel_requested = False
            self.trigger_count += 1
            self.latest_error = None
            self.latest_progress = None
            payload = dict(self.context)
            generation = self.trigger_count

        thread = threading.Thread(
            target=self._run_capture,
            args=(payload, generation),
            daemon=True,
        )
        thread.start()
        return self.status()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            self.cancel_requested = True
            self.running = False
            self.latest_error = "cancel_requested"
            self.updated_at = time.time()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hotkey": self.display_hotkey,
                "raw_hotkey": self.hotkey,
                "running": self.running,
                "cancel_requested": self.cancel_requested,
                "latest_result": self.latest_result,
                "latest_progress": self.latest_progress,
                "latest_error": self.latest_error,
                "trigger_count": self.trigger_count,
                "latest_result_generation": self.latest_result_generation,
                "latest_progress_generation": self.latest_progress_generation,
                "updated_at": self.updated_at,
                "context": dict(self.context),
                "listener_active": self._listener is not None,
            }

    def _run_capture(self, payload: dict[str, Any], generation: int) -> None:
        try:
            if not payload.get("user_input"):
                result = {
                    "status": "missing_user_input",
                    "reply": {
                        "should_reply": False,
                        "content": "请先输入你需要我帮什么，比如“她回我哦，我该怎么回？”",
                        "reason": "missing_user_input",
                    },
                    "error": "missing_user_input",
                }
            else:
                self._pre_capture_for_hotkey(payload, generation)
                result = self.agent_factory().process(payload)
            with self._lock:
                if self.cancel_requested or generation != self.trigger_count:
                    return
                self.latest_result = result
                self.latest_result_generation = generation
                self.latest_error = None
                self.updated_at = time.time()
        except Exception as exc:
            with self._lock:
                if generation == self.trigger_count and not self.cancel_requested:
                    self.latest_error = str(exc)
                    self.updated_at = time.time()
        finally:
            with self._lock:
                if generation == self.trigger_count:
                    self.running = False

    def _pre_capture_for_hotkey(self, payload: dict[str, Any], generation: int) -> None:
        """Capture immediately after hotkey trigger and publish progress."""
        if payload.get("screenshot_base64"):
            payload["screenshot_captured"] = True
            payload["screenshot_status"] = payload.get("screenshot_status") or "uploaded"
            self._publish_progress(
                {
                    "type": "screenshot_captured",
                    "message": "截图已完成。",
                    "screenshot_status": payload["screenshot_status"],
                },
                generation,
            )
            return

        screenshot_region = payload.get("screenshot_region")
        if not screenshot_region:
            return

        try:
            screenshot_base64 = capture_region_as_base64(screenshot_region)
        except Exception as exc:
            payload["pre_capture_status"] = "failed"
            payload["pre_capture_error"] = str(exc)
            self._publish_progress(
                {
                    "type": "screenshot_failed",
                    "message": "截图失败。",
                    "error": str(exc),
                },
                generation,
            )
            return

        payload["screenshot_base64"] = screenshot_base64
        payload["screenshot_captured"] = True
        payload["screenshot_status"] = "captured"
        payload["pre_capture_status"] = "success"
        payload["pre_capture_error"] = None
        self._publish_progress(
            {
                "type": "screenshot_captured",
                "message": "截图已完成。",
                "screenshot_status": "captured",
            },
            generation,
        )

    def _publish_progress(self, progress: dict[str, Any], generation: int) -> None:
        with self._lock:
            if self.cancel_requested or generation != self.trigger_count:
                return
            self.latest_progress = dict(progress)
            self.latest_progress_generation = generation
            self.updated_at = time.time()
