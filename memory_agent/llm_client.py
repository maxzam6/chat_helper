from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Generic model client interface used by GraphMemoryAgent."""

    @abstractmethod
    def generate_json(self, task: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one logical model task and return structured JSON."""


class MockLLMClient(BaseLLMClient):
    """Local deterministic model mock for development and tests."""

    def generate_json(self, task: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if task == "intent_classifier":
            return self._intent_classifier(inputs)
        if task == "reply":
            return self._reply(inputs)
        if task == "ocr":
            return self._ocr(inputs)
        if task == "retrieval_query":
            return self._retrieval_query(inputs)
        if task == "learning":
            return self._learning(inputs)
        return {}

    def _intent_classifier(self, inputs: dict[str, Any]) -> dict[str, str]:
        user_input = str(inputs.get("user_input", ""))
        if any(token in user_input for token in ("短一点", "自然一点", "太油腻", "不要这么", "改一下", "换个说法")):
            intent = "revise_reply"
        elif any(token in user_input for token in ("记一下", "画像", "其实她", "其实他", "她不是", "他不是", "不喜欢", "喜欢", "平时")):
            intent = "profile_update"
        elif any(token in user_input for token in ("怎么回", "回复", "她说", "他说", "聊天", "帮我看看", "该不该")):
            intent = "reply_advice"
        else:
            intent = "general_question"
        return {"intent": intent, "input_summary": user_input}

    def _reply(self, inputs: dict[str, Any]) -> dict[str, Any]:
        intent = inputs.get("intent")
        input_summary = str(inputs.get("input_summary", ""))
        if intent == "revise_reply":
            content = "可以，改成更短、更自然一点。"
            reason = "mock revised previous reply"
        elif intent == "reply_advice":
            content = "可以轻松回一句：'哈哈，感觉你今天有点累，那先不打扰你啦。'"
            reason = "mock reply advice"
        elif intent == "profile_update":
            content = "已更新画像。"
            reason = "mock profile update confirmation"
        else:
            content = f"这是一个本地 mock 回答：{input_summary}"
            reason = "mock general answer"
        return {"reply": {"should_reply": True, "content": content, "reason": reason}}

    def _ocr(self, inputs: dict[str, Any]) -> dict[str, Any]:
        chat_context = inputs.get("chat_context")
        if isinstance(chat_context, dict) and chat_context:
            context = chat_context
        else:
            context = {
                "recent_messages": [
                    {"role": "user", "content": "哦"},
                    {"role": "me", "content": str(inputs.get("user_input", "怎么回比较自然？"))},
                ],
                "previous_recent_messages": [],
            }
        return {
            "chat_context": context,
            "working_memory_observations": [
                {
                    "content": "The current chat may need a concise, low-pressure response.",
                    "confidence": 0.72,
                }
            ],
        }

    def _retrieval_query(self, inputs: dict[str, Any]) -> dict[str, str]:
        input_summary = str(inputs.get("input_summary", "")).strip()
        if input_summary:
            return {"retrieval_query": input_summary}
        chat_context = inputs.get("chat_context") or {}
        recent_messages = chat_context.get("recent_messages") or []
        text = " ".join(str(message.get("content", "")) for message in recent_messages)
        return {"retrieval_query": text.strip()}

    def _learning(self, inputs: dict[str, Any]) -> dict[str, Any]:
        intent = inputs.get("intent")
        if intent == "profile_update":
            content = "The user says the target usually speaks little, which should not be overread as coldness."
            memory_type = "profile_note"
        else:
            content = "The target may prefer concise, low-pressure replies in this context."
            memory_type = "conversation_style"
        return {
            "memory_updates": [
                {
                    "memory_type": memory_type,
                    "content": content,
                    "confidence": 0.82,
                    "has_conflict": False,
                }
            ],
            "memory_reviews": [],
            "changed_summary": "Memory profile updated from latest context.",
        }


class LLMClient(MockLLMClient):
    """Default local client; replace with a real provider later."""
