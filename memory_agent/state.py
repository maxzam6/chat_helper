from __future__ import annotations

from typing import Any, Literal, TypedDict


Intent = Literal["general_question", "revise_reply", "reply_advice", "profile_update"]


class AgentState(TypedDict, total=False):
    me_id: str | None
    current_user_id: str | None
    active_user_id: str | None

    user_input: str
    intent: Intent
    input_summary: str
    task_list: list[str]
    current_task: str | None
    completed_tasks: list[str]
    task_results: list[dict[str, Any]]

    reply: dict[str, Any]
    last_reply: dict[str, Any] | None
    session_state: dict[str, Any] | None

    screenshot_path: str | None
    screenshot_base64: str | None
    screenshot_region: dict[str, Any] | None
    screenshot_captured: bool
    screenshot_status: str | None
    pre_capture_status: str | None
    pre_capture_error: str | None
    is_valid_chat_window: bool | None
    validation_reason: str | None
    vision_error: str | None
    recognized_user_id: str | None
    user_id_change_detected: bool | None

    chat_context: dict[str, Any]
    chat_text: str

    working_memory: list[dict[str, Any]]
    working_memory_observations: list[dict[str, Any]]

    retrieval_query: str
    last_retrieval_query: str | None
    query_similarity: float
    reuse_cache: bool

    semantic_results: list[dict[str, Any]]
    relevant_memories: list[dict[str, Any]]
    active_memory_cache: dict[str, Any]

    memory_updates: list[dict[str, Any]]
    memory_reviews: list[dict[str, Any]]
    changed_summary: str | None

    dirty_memories: list[dict[str, Any]]
    saved_memory_ids: list[int]
    reviewed_memories: list[dict[str, Any]]
    discarded_memory_ids: list[int]
    sync_errors: list[dict[str, Any]]
    context_switch_sync_result: dict[str, Any] | None
    session_state_saved: bool | None
    reason: str | None
    user_id_suggestions: list[dict[str, Any]]

    status: str
    error: str | None
