from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FilterResult:
    should_process: bool
    reason: str | None = None


class InputFilter:
    """输入过滤器。

    当前只做任务要求中的两件事：
    1. recent_messages 为空时跳过
    2. recent_messages 和 previous_recent_messages 完全重复时跳过
    """

    def check(self, payload: dict[str, Any]) -> FilterResult:
        chat_context = payload.get("chat_context") or {}
        recent_messages = chat_context.get("recent_messages") or []
        previous_recent_messages = chat_context.get("previous_recent_messages") or []

        if recent_messages == []:
            return FilterResult(False, "empty_recent_messages")

        if self._signature(recent_messages) == self._signature(previous_recent_messages):
            return FilterResult(False, "duplicate_recent_messages")

        return FilterResult(True)

    def _signature(self, messages: list[dict[str, Any]]) -> str:
        # 把整段消息转成稳定字符串，用于比较两段聊天是否完全一致。
        return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
