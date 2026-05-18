from __future__ import annotations

import json
from typing import Any


def messages_to_chat_text(messages: list[dict[str, Any]]) -> str:
    """Convert recent_messages into plain text for Dify variables."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role or content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_dify_inputs(
    payload: dict[str, Any],
    memory: list[str] | None = None,
    working_memory: dict[str, Any] | None = None,
    relevant_memories: list[dict[str, Any]] | None = None,
    stage: str = "learning",
) -> dict[str, Any]:
    """Build Dify Workflow inputs.

    Python only packages data. Prompting, reply generation, learning, and working
    memory summarization belong in Dify nodes.
    """
    user_id = payload["user_id"]
    chat_context = payload.get("chat_context") or {}
    recent_messages = chat_context.get("recent_messages") or []

    return {
        "stage": stage,
        "chatText": messages_to_chat_text(recent_messages),
        "info": {
            "user_id": user_id,
            "memory": memory or [],
            "working_memory": working_memory,
            "relevant_memories": relevant_memories or [],
            "chat_context": chat_context,
        },
    }


def extract_retrieval_query(dify_output: dict[str, Any]) -> str:
    """Extract retrieval_query from direct output, data.outputs, or result JSON."""
    if isinstance(dify_output.get("retrieval_query"), str):
        return dify_output["retrieval_query"].strip()

    outputs = _outputs(dify_output)
    if isinstance(outputs.get("retrieval_query"), str):
        return outputs["retrieval_query"].strip()

    result = parse_dify_result(dify_output)
    if isinstance(result.get("retrieval_query"), str):
        return result["retrieval_query"].strip()

    return ""


def extract_memory_update(dify_output: dict[str, Any]) -> dict[str, Any]:
    """Extract legacy single memory_update."""
    if isinstance(dify_output.get("memory_update"), dict):
        return dify_output["memory_update"]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get("memory_update"), dict):
        return outputs["memory_update"]

    result = parse_dify_result(dify_output)
    if isinstance(result.get("memory_update"), dict):
        return result["memory_update"]

    return {}


def extract_memory_updates(dify_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract one or more memory updates from Dify output."""
    if isinstance(dify_output.get("memory_updates"), list):
        return [item for item in dify_output["memory_updates"] if isinstance(item, dict)]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get("memory_updates"), list):
        return [item for item in outputs["memory_updates"] if isinstance(item, dict)]

    result = parse_dify_result(dify_output)
    if isinstance(result.get("memory_updates"), list):
        return [item for item in result["memory_updates"] if isinstance(item, dict)]

    single_update = extract_memory_update(dify_output)
    return [single_update] if single_update else []


def extract_memory_review(dify_output: dict[str, Any]) -> dict[str, Any]:
    """Extract legacy single memory_review result."""
    if isinstance(dify_output.get("memory_review"), dict):
        return dify_output["memory_review"]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get("memory_review"), dict):
        return outputs["memory_review"]

    result = parse_dify_result(dify_output)
    if isinstance(result.get("memory_review"), dict):
        return result["memory_review"]

    if "confidence" in dify_output:
        return dify_output

    return {}


def extract_memory_reviews(dify_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract memory review results from Dify output."""
    if isinstance(dify_output.get("memory_reviews"), list):
        return [item for item in dify_output["memory_reviews"] if isinstance(item, dict)]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get("memory_reviews"), list):
        return [item for item in outputs["memory_reviews"] if isinstance(item, dict)]

    result = parse_dify_result(dify_output)
    if isinstance(result.get("memory_reviews"), list):
        return [item for item in result["memory_reviews"] if isinstance(item, dict)]

    single_review = extract_memory_review(dify_output)
    return [single_review] if single_review else []


def extract_updated_working_memory(dify_output: dict[str, Any]) -> dict[str, Any]:
    """Extract updated_working_memory from Dify output."""
    if isinstance(dify_output.get("updated_working_memory"), dict):
        return dify_output["updated_working_memory"]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get("updated_working_memory"), dict):
        return outputs["updated_working_memory"]

    result = parse_dify_result(dify_output)
    if isinstance(result.get("updated_working_memory"), dict):
        return result["updated_working_memory"]

    return {}


def extract_reply(dify_output: dict[str, Any]) -> dict[str, Any]:
    """Extract reply object from Dify output."""
    if isinstance(dify_output.get("reply"), dict):
        return dify_output["reply"]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get("reply"), dict):
        return outputs["reply"]

    result = parse_dify_result(dify_output)
    if isinstance(result.get("reply"), dict):
        return result["reply"]

    return {}


def extract_intent_result(dify_output: dict[str, Any]) -> dict[str, str]:
    """Extract intent classification output."""
    result = _extract_dict_field(dify_output, "intent_result")
    if not result:
        result = {
            "intent": _extract_string_field(dify_output, "intent"),
            "input_summary": _extract_string_field(dify_output, "input_summary"),
        }
    return {
        "intent": result.get("intent") or "general_question",
        "input_summary": result.get("input_summary") or "",
    }


def extract_chat_context(dify_output: dict[str, Any]) -> dict[str, Any]:
    """Extract chat_context from OCR/Vision output."""
    return _extract_dict_field(dify_output, "chat_context")


def extract_working_memory_observations(dify_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract working memory observation list from Dify output."""
    observations = _extract_list_field(dify_output, "working_memory_observations")
    return [item for item in observations if isinstance(item, dict)]


def extract_changed_summary(dify_output: dict[str, Any]) -> str:
    """Extract profile/memory changed summary."""
    return _extract_string_field(dify_output, "changed_summary")


def parse_dify_result(dify_output: dict[str, Any]) -> dict[str, Any]:
    """Parse Dify End-node result JSON without raising on bad output."""
    result = dify_output.get("result")
    if result is None:
        result = _outputs(dify_output).get("result")

    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return {}

    cleaned = _strip_markdown_json_block(result)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _outputs(dify_output: dict[str, Any]) -> dict[str, Any]:
    data = dify_output.get("data") or {}
    outputs = data.get("outputs") or {}
    return outputs if isinstance(outputs, dict) else {}


def _extract_dict_field(dify_output: dict[str, Any], key: str) -> dict[str, Any]:
    if isinstance(dify_output.get(key), dict):
        return dify_output[key]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get(key), dict):
        return outputs[key]

    result = parse_dify_result(dify_output)
    if isinstance(result.get(key), dict):
        return result[key]

    return {}


def _extract_list_field(dify_output: dict[str, Any], key: str) -> list[Any]:
    if isinstance(dify_output.get(key), list):
        return dify_output[key]

    outputs = _outputs(dify_output)
    if isinstance(outputs.get(key), list):
        return outputs[key]

    result = parse_dify_result(dify_output)
    if isinstance(result.get(key), list):
        return result[key]

    return []


def _extract_string_field(dify_output: dict[str, Any], key: str) -> str:
    if isinstance(dify_output.get(key), str):
        return dify_output[key].strip()

    outputs = _outputs(dify_output)
    if isinstance(outputs.get(key), str):
        return outputs[key].strip()

    result = parse_dify_result(dify_output)
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
