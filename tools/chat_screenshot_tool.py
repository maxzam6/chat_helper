from __future__ import annotations

from typing import Any

from memory_agent.models import (
    extract_chat_context,
    extract_working_memory_observations,
    messages_to_chat_text,
)
from memory_agent.vision_llm_client import BaseVisionLLMClient

from .screenshot_tool import capture_region_as_base64


class ChatScreenshotParseError(RuntimeError):
    """Raised when screen capture succeeds but Vision LLM parsing fails."""

    def __init__(self, message: str, screenshot_base64: str) -> None:
        super().__init__(message)
        self.screenshot_base64 = screenshot_base64


def capture_and_parse_chat_tool(
    screenshot_region: dict[str, Any],
    user_input: str,
    vision_llm_client: BaseVisionLLMClient,
) -> dict[str, Any]:
    """Capture a chat area and parse it with the independent Vision LLM."""
    screenshot_base64 = capture_region_as_base64(screenshot_region)
    try:
        output = vision_llm_client.parse_chat_screenshot(screenshot_base64, user_input)
    except Exception as exc:
        raise ChatScreenshotParseError(f"vision_parse_failed:{exc}", screenshot_base64) from exc
    chat_context = extract_chat_context(output)
    recent_messages = chat_context.get("recent_messages") or []
    is_valid = output.get("is_valid_chat_window")
    if not isinstance(is_valid, bool):
        is_valid = _has_valid_recent_messages(recent_messages)

    return {
        "screenshot_base64": screenshot_base64,
        "screenshot_captured": True,
        "screenshot_status": "captured",
        "is_valid_chat_window": is_valid,
        "validation_reason": str(output.get("validation_reason") or ""),
        "recognized_user_id": _clean_optional_string(output.get("recognized_user_id")),
        "chat_context": chat_context,
        "chat_text": messages_to_chat_text(recent_messages),
        "working_memory_observations": extract_working_memory_observations(output),
    }


def parse_uploaded_chat_screenshot(
    screenshot_base64: str,
    user_input: str,
    vision_llm_client: BaseVisionLLMClient,
) -> dict[str, Any]:
    """Parse an already uploaded screenshot without taking a desktop capture."""
    try:
        output = vision_llm_client.parse_chat_screenshot(screenshot_base64, user_input)
    except Exception as exc:
        raise ChatScreenshotParseError(f"vision_parse_failed:{exc}", screenshot_base64) from exc
    chat_context = extract_chat_context(output)
    recent_messages = chat_context.get("recent_messages") or []
    is_valid = output.get("is_valid_chat_window")
    if not isinstance(is_valid, bool):
        is_valid = _has_valid_recent_messages(recent_messages)
    return {
        "screenshot_base64": screenshot_base64,
        "screenshot_captured": True,
        "screenshot_status": "uploaded",
        "is_valid_chat_window": is_valid,
        "validation_reason": str(output.get("validation_reason") or ""),
        "recognized_user_id": _clean_optional_string(output.get("recognized_user_id")),
        "chat_context": chat_context,
        "chat_text": messages_to_chat_text(recent_messages),
        "working_memory_observations": extract_working_memory_observations(output),
    }


def _has_valid_recent_messages(recent_messages: Any) -> bool:
    if not isinstance(recent_messages, list) or not recent_messages:
        return False
    return any(
        isinstance(message, dict)
        and str(message.get("role", "")).strip()
        and str(message.get("content", "")).strip()
        for message in recent_messages
    )


def _clean_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
