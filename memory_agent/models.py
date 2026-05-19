from __future__ import annotations

import json
from typing import Any


def messages_to_chat_text(messages: list[dict[str, Any]]) -> str:
    """Convert recent_messages into plain text for model inputs."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role or content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def parse_llm_json(value: Any) -> dict[str, Any]:
    """Parse generic model JSON output; return {} on malformed content."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    cleaned = _strip_markdown_json_block(value)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_intent_result(output: dict[str, Any]) -> dict[str, Any]:
    result = _extract_dict_field(output, "intent_result")
    if not result:
        result = {
            "intent": _extract_string_field(output, "intent"),
            "intents": _extract_list_field(output, "intents"),
            "input_summary": _extract_string_field(output, "input_summary"),
        }
    intent = result.get("intent") or "general_question"
    raw_intents = result.get("intents")
    intents = raw_intents if isinstance(raw_intents, list) else []
    intents = [str(item) for item in intents if isinstance(item, str) and item.strip()]
    if not intents:
        intents = [intent]
    return {
        "intent": intent,
        "intents": intents,
        "input_summary": result.get("input_summary") or "",
    }


def extract_reply(output: dict[str, Any]) -> dict[str, Any]:
    return _extract_dict_field(output, "reply")


def extract_chat_context(output: dict[str, Any]) -> dict[str, Any]:
    return _extract_dict_field(output, "chat_context")


def extract_working_memory_observations(output: dict[str, Any]) -> list[dict[str, Any]]:
    observations = _extract_list_field(output, "working_memory_observations")
    return [item for item in observations if isinstance(item, dict)]


def extract_retrieval_query(output: dict[str, Any]) -> str:
    return _extract_string_field(output, "retrieval_query")


def extract_memory_update(output: dict[str, Any]) -> dict[str, Any]:
    return _extract_dict_field(output, "memory_update")


def extract_memory_updates(output: dict[str, Any]) -> list[dict[str, Any]]:
    updates = _extract_list_field(output, "memory_updates")
    if updates:
        return [item for item in updates if isinstance(item, dict)]
    single_update = extract_memory_update(output)
    return [single_update] if single_update else []


def extract_memory_review(output: dict[str, Any]) -> dict[str, Any]:
    review = _extract_dict_field(output, "memory_review")
    if review:
        return review
    if "confidence" in output:
        return output
    result = _result_payload(output)
    return result if "confidence" in result else {}


def extract_memory_reviews(output: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = _extract_list_field(output, "memory_reviews")
    if reviews:
        return [item for item in reviews if isinstance(item, dict)]
    single_review = extract_memory_review(output)
    return [single_review] if single_review else []


def extract_updated_working_memory(output: dict[str, Any]) -> dict[str, Any]:
    return _extract_dict_field(output, "updated_working_memory")


def extract_changed_summary(output: dict[str, Any]) -> str:
    return _extract_string_field(output, "changed_summary")


def _result_payload(output: dict[str, Any]) -> dict[str, Any]:
    if "result" in output:
        return parse_llm_json(output["result"])
    outputs = _outputs(output)
    if "result" in outputs:
        return parse_llm_json(outputs["result"])
    return {}


def _outputs(output: dict[str, Any]) -> dict[str, Any]:
    data = output.get("data") or {}
    outputs = data.get("outputs") or {}
    return outputs if isinstance(outputs, dict) else {}


def _extract_dict_field(output: dict[str, Any], key: str) -> dict[str, Any]:
    if isinstance(output.get(key), dict):
        return output[key]

    outputs = _outputs(output)
    if isinstance(outputs.get(key), dict):
        return outputs[key]

    result = _result_payload(output)
    if isinstance(result.get(key), dict):
        return result[key]

    return {}


def _extract_list_field(output: dict[str, Any], key: str) -> list[Any]:
    if isinstance(output.get(key), list):
        return output[key]

    outputs = _outputs(output)
    if isinstance(outputs.get(key), list):
        return outputs[key]

    result = _result_payload(output)
    if isinstance(result.get(key), list):
        return result[key]

    return []


def _extract_string_field(output: dict[str, Any], key: str) -> str:
    if isinstance(output.get(key), str):
        return output[key].strip()

    outputs = _outputs(output)
    if isinstance(outputs.get(key), str):
        return outputs[key].strip()

    result = _result_payload(output)
    if isinstance(result.get(key), str):
        return result[key].strip()

    return ""


def _strip_markdown_json_block(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            first_line = lines[0].strip()
            if first_line in {"```", "```json", "```JSON"}:
                lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned
