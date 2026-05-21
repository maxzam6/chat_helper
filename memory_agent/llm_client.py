from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .models import parse_llm_json


class BaseLLMClient(ABC):
    """Generic model client interface used by GraphMemoryAgent."""

    @abstractmethod
    def generate_json(self, task: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one logical model task and return structured JSON."""


class MockLLMClient(BaseLLMClient):
    """Local deterministic model mock for tests only."""

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

    def _intent_classifier(self, inputs: dict[str, Any]) -> dict[str, Any]:
        user_input = str(inputs.get("user_input", ""))
        intents: list[str] = []
        if any(token in user_input for token in ("是什么", "为什么", "怎么做", "LangGraph", "解释")):
            intents.append("general_question")
        if any(token in user_input for token in ("短一点", "自然一点", "太油腻", "不要这么", "改一下", "换个说法")):
            intents.append("revise_reply")
        if any(token in user_input for token in ("怎么回", "回复", "她说", "他说", "聊天", "帮我看看", "该不该")):
            intents.append("reply_advice")
        if any(token in user_input for token in ("记一下", "画像", "其实她", "其实他", "她不是", "他不是", "不喜欢", "喜欢", "平时")):
            intents.append("profile_update")
        if not intents:
            intents.append("general_question")
        return {"intent": intents[0], "intents": intents, "input_summary": user_input}

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


class LLMClient(BaseLLMClient):
    """OpenAI-compatible JSON model client for the main GraphMemoryAgent path."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        response_format: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or None
        self.model = model or os.getenv("LLM_MODEL")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.2"))
        )
        self.response_format = (
            response_format
            if response_format is not None
            else os.getenv("LLM_RESPONSE_FORMAT", "json_object")
        )

        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is required when using the real LLMClient.")
        if not self.model:
            raise RuntimeError("LLM_MODEL is required when using the real LLMClient.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.vision_api_key = os.getenv("VISION_LLM_API_KEY") or self.api_key
        self.vision_base_url = os.getenv("VISION_LLM_BASE_URL") or self.base_url
        self.vision_model = os.getenv("VISION_LLM_MODEL") or self.model
        self.vision_client = OpenAI(
            api_key=self.vision_api_key,
            base_url=self.vision_base_url,
        )

    def generate_json(self, task: str, inputs: dict[str, Any]) -> dict[str, Any]:
        system_prompt = self._load_prompt(task)
        if task == "ocr" and inputs.get("screenshot_base64"):
            return self._generate_vision_json(system_prompt, inputs)

        user_content = json.dumps(inputs, ensure_ascii=False, indent=2)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
        }
        if self.response_format and self.response_format.lower() != "none":
            request_kwargs["response_format"] = {"type": self.response_format}

        response = self.client.chat.completions.create(**request_kwargs)
        text = response.choices[0].message.content or ""
        return parse_llm_json(text)

    def _generate_vision_json(
        self,
        system_prompt: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the OCR/vision task with an image message when a screenshot exists."""
        screenshot = str(inputs.get("screenshot_base64") or "")
        if not screenshot.startswith("data:image/"):
            screenshot = f"data:image/png;base64,{screenshot}"

        text_inputs = dict(inputs)
        text_inputs["screenshot_base64"] = "[image attached]"
        user_text = json.dumps(text_inputs, ensure_ascii=False, indent=2)

        request_kwargs: dict[str, Any] = {
            "model": self.vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": screenshot}},
                    ],
                },
            ],
            "temperature": self.temperature,
        }
        if self.response_format and self.response_format.lower() != "none":
            request_kwargs["response_format"] = {"type": self.response_format}

        response = self.vision_client.chat.completions.create(**request_kwargs)
        text = response.choices[0].message.content or ""
        return parse_llm_json(text)

    def _load_prompt(self, task: str) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / f"{task}.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found for task: {task}")

        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty for task: {task}")
        return prompt
