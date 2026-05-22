from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .models import parse_llm_json


class BaseVisionLLMClient(ABC):
    """Interface for screenshot/chat-window parsing models."""

    @abstractmethod
    def parse_chat_screenshot(
        self,
        image_base64: str,
        user_input: str,
    ) -> dict[str, Any]:
        """Parse one chat screenshot and return structured JSON."""


class VisionLLMClient(BaseVisionLLMClient):
    """OpenAI-compatible multimodal client used only for Vision/OCR tasks."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        response_format: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("VISION_API_KEY")
        self.base_url = base_url or os.getenv("VISION_BASE_URL") or None
        self.model = model or os.getenv("VISION_MODEL")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("VISION_TEMPERATURE", "0.0"))
        )
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("VISION_TIMEOUT", "60"))
        )
        self.response_format = (
            response_format
            if response_format is not None
            else os.getenv("VISION_RESPONSE_FORMAT", "json_object")
        )

        if not self.api_key:
            raise RuntimeError("VISION_API_KEY is required when using VisionLLMClient.")
        if not self.model:
            raise RuntimeError("VISION_MODEL is required when using VisionLLMClient.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def parse_chat_screenshot(
        self,
        image_base64: str,
        user_input: str,
    ) -> dict[str, Any]:
        image_url = self._normalize_image_data_url(image_base64)
        text_inputs = json.dumps(
            {"user_input": user_input},
            ensure_ascii=False,
            indent=2,
        )
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._load_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_inputs},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": self.temperature,
        }
        if self.response_format and self.response_format.lower() != "none":
            request_kwargs["response_format"] = {"type": self.response_format}

        response = self.client.chat.completions.create(**request_kwargs)
        text = response.choices[0].message.content or ""
        return parse_llm_json(text)

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "vision_chat.md"
        if not prompt_path.exists():
            raise FileNotFoundError("Vision prompt file not found: vision_chat.md")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError("Vision prompt file is empty: vision_chat.md")
        return prompt

    def _normalize_image_data_url(self, image_base64: str) -> str:
        image = image_base64.strip()
        if image.startswith("data:image/"):
            return image
        return f"data:image/png;base64,{image}"


class MockVisionLLMClient(BaseVisionLLMClient):
    """Deterministic test double. Do not use as a production default."""

    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def parse_chat_screenshot(
        self,
        image_base64: str,
        user_input: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "image_base64": image_base64,
                "user_input": user_input,
            }
        )
        if self.output is not None:
            return self.output
        return {
            "is_valid_chat_window": True,
            "validation_reason": "mock_valid_chat_window",
            "recognized_user_id": None,
            "chat_context": {
                "recent_messages": [
                    {"role": "target", "content": "哦"},
                    {"role": "me", "content": user_input or "怎么回？"},
                ],
                "previous_recent_messages": [],
            },
            "working_memory_observations": [
                {
                    "content": "当前截图中聊天信息较少，适合低压力回应。",
                    "confidence": 0.72,
                }
            ],
        }
