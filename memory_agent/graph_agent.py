from __future__ import annotations

from typing import Any, Callable

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    START = "__start__"
    END = "__end__"

    class StateGraph:  # type: ignore[no-redef]
        """Small local fallback matching the LangGraph methods used here."""

        def __init__(self, state_type: Any) -> None:
            self.nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
            self.edges: dict[str, list[str]] = {}
            self.conditional_edges: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]] = {}

        def add_node(self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
            self.nodes[name] = fn

        def add_edge(self, source: str, target: str) -> None:
            self.edges.setdefault(source, []).append(target)

        def add_conditional_edges(
            self,
            source: str,
            router: Callable[[dict[str, Any]], str],
            mapping: dict[str, str],
        ) -> None:
            self.conditional_edges[source] = (router, mapping)

        def compile(self) -> "_CompiledFallbackGraph":
            return _CompiledFallbackGraph(self)

    class _CompiledFallbackGraph:
        def __init__(self, graph: StateGraph) -> None:
            self.graph = graph

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            current = self.graph.edges[START][0]
            while current != END:
                update = self.graph.nodes[current](state)
                if update:
                    state.update(update)
                if current in self.graph.conditional_edges:
                    router, mapping = self.graph.conditional_edges[current]
                    current = mapping[router(state)]
                else:
                    current = self.graph.edges.get(current, [END])[0]
            return state

from .active_memory_cache import ActiveMemoryCache
from .input_filter import InputFilter
from .llm_client import BaseLLMClient, MockLLMClient
from .memory_store import MemoryStore, classify_memory_status
from .models import (
    extract_changed_summary,
    extract_chat_context,
    extract_intent_result,
    extract_memory_reviews,
    extract_memory_updates,
    extract_reply,
    extract_retrieval_query,
    extract_working_memory_observations,
    messages_to_chat_text,
)
from .semantic_retriever import SemanticRetriever
from .state import AgentState


TASK_PRIORITY = ["general_question", "revise_reply", "reply_advice", "profile_update"]
ALLOWED_INTENTS = set(TASK_PRIORITY)


class GraphMemoryAgent:
    """LangGraph-based memory agent using a generic model client."""

    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: BaseLLMClient | None = None,
        input_filter: InputFilter | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        active_memory_cache: ActiveMemoryCache | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.llm_client = llm_client or MockLLMClient()
        self.input_filter = input_filter or InputFilter()
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        self.active_memory_cache = active_memory_cache or ActiveMemoryCache()
        self.app = self.build_graph().compile()

    def build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("select_next_task", self.select_next_task)
        graph.add_node("reply_general", self.reply_general)
        graph.add_node("load_session_state", self.load_session_state)
        graph.add_node("check_last_reply", self.check_last_reply)
        graph.add_node("reply_missing_context", self.reply_missing_context)
        graph.add_node("revise_reply", self.revise_reply)
        graph.add_node("check_user_context", self.check_user_context)
        graph.add_node("reply_missing_user_id", self.reply_missing_user_id)
        graph.add_node("take_screenshot_or_mock", self.take_screenshot_or_mock)
        graph.add_node("ocr_vision_llm", self.ocr_vision_llm)
        graph.add_node("input_filter", self.input_filter_node)
        graph.add_node("reply_input_skipped", self.reply_input_skipped)
        graph.add_node("update_working_memory", self.update_working_memory)
        graph.add_node("retrieval_query_llm", self.retrieval_query_llm)
        graph.add_node("query_similarity_check", self.query_similarity_check)
        graph.add_node("reuse_cache", self.reuse_cache)
        graph.add_node("retrieve_and_build_cache", self.retrieve_and_build_cache)
        graph.add_node("after_cache", self.noop)
        graph.add_node("reply_advice_llm", self.reply_advice_llm)
        graph.add_node("learning_llm", self.learning_llm)
        graph.add_node("update_cache_from_learning", self.update_cache_from_learning)
        graph.add_node("sync_dirty_memory", self.sync_dirty_memory)
        graph.add_node("profile_update_confirm_reply", self.profile_update_confirm_reply)
        graph.add_node("save_session_state", self.save_session_state)
        graph.add_node("mark_task_done", self.mark_task_done)

        graph.add_edge(START, "classify_intent")
        graph.add_edge("classify_intent", "select_next_task")
        graph.add_conditional_edges(
            "select_next_task",
            self.route_by_intent,
            {
                "general_question": "reply_general",
                "revise_reply": "load_session_state",
                "reply_advice": "check_user_context",
                "profile_update": "check_user_context",
                "done": END,
            },
        )
        graph.add_edge("reply_general", "save_session_state")
        graph.add_edge("load_session_state", "check_last_reply")
        graph.add_conditional_edges(
            "check_last_reply",
            self.route_last_reply,
            {"has_last_reply": "revise_reply", "no_last_reply": "reply_missing_context"},
        )
        graph.add_edge("reply_missing_context", "save_session_state")
        graph.add_edge("revise_reply", "save_session_state")
        graph.add_conditional_edges(
            "check_user_context",
            self.route_after_user_context,
            {
                "missing_user_id": "reply_missing_user_id",
                "reply_advice": "take_screenshot_or_mock",
                "profile_update": "retrieval_query_llm",
            },
        )
        graph.add_edge("reply_missing_user_id", "save_session_state")
        graph.add_edge("take_screenshot_or_mock", "ocr_vision_llm")
        graph.add_edge("ocr_vision_llm", "input_filter")
        graph.add_conditional_edges(
            "input_filter",
            self.route_input_filter,
            {"skipped": "reply_input_skipped", "ok": "update_working_memory"},
        )
        graph.add_edge("reply_input_skipped", "save_session_state")
        graph.add_edge("update_working_memory", "retrieval_query_llm")
        graph.add_edge("retrieval_query_llm", "query_similarity_check")
        graph.add_conditional_edges(
            "query_similarity_check",
            self.route_cache,
            {"reuse_cache": "reuse_cache", "retrieve_and_build_cache": "retrieve_and_build_cache"},
        )
        graph.add_edge("reuse_cache", "after_cache")
        graph.add_edge("retrieve_and_build_cache", "after_cache")
        graph.add_conditional_edges(
            "after_cache",
            self.route_after_cache,
            {"reply_advice": "reply_advice_llm", "profile_update": "learning_llm"},
        )
        graph.add_edge("reply_advice_llm", "learning_llm")
        graph.add_edge("learning_llm", "update_cache_from_learning")
        graph.add_edge("update_cache_from_learning", "sync_dirty_memory")
        graph.add_conditional_edges(
            "sync_dirty_memory",
            self.route_after_sync,
            {"reply_advice": "save_session_state", "profile_update": "profile_update_confirm_reply"},
        )
        graph.add_edge("profile_update_confirm_reply", "save_session_state")
        graph.add_edge("save_session_state", "mark_task_done")
        graph.add_edge("mark_task_done", "select_next_task")
        return graph

    def process(self, payload: dict[str, Any], return_full_state: bool = False) -> dict[str, Any]:
        self.memory_store.init_db()
        state = self.app.invoke(self._initial_state(payload))
        if return_full_state:
            return state
        return self._public_result(state)

    def classify_intent(self, state: AgentState) -> dict[str, Any]:
        output = self.llm_client.generate_json(
            task="intent_classifier",
            inputs={
                "user_input": state.get("user_input", ""),
                "me_id": state.get("me_id"),
                "current_user_id": state.get("current_user_id"),
            },
        )
        result = extract_intent_result(output)
        task_list = self._normalize_task_list(result.get("intents") or [result.get("intent")])
        intent = task_list[0]
        return {
            "intent": intent,
            "current_task": None,
            "task_list": task_list,
            "completed_tasks": [],
            "task_results": [],
            "input_summary": result.get("input_summary") or state.get("user_input", ""),
            "status": "intent_classified",
        }

    def select_next_task(self, state: AgentState) -> dict[str, Any]:
        completed_tasks = set(state.get("completed_tasks", []))
        for task in state.get("task_list", []):
            if task in completed_tasks:
                continue
            return {
                "intent": task,
                "current_task": task,
                "status": "task_selected",
                "reply": {},
                "retrieval_query": "",
                "query_similarity": 0.0,
                "reuse_cache": False,
                "semantic_results": [],
                "relevant_memories": [],
                "memory_updates": [],
                "memory_reviews": [],
                "changed_summary": None,
                "dirty_memories": [],
                "saved_memory_ids": [],
                "reviewed_memories": [],
                "discarded_memory_ids": [],
                "sync_errors": [],
                "session_state_saved": None,
                "error": None,
            }
        return {}

    def reply_general(self, state: AgentState) -> dict[str, Any]:
        session_state = self.memory_store.get_session_state(state.get("me_id") or "default", "global")
        output = self.llm_client.generate_json(
            task="reply",
            inputs={
                "intent": "general_question",
                "user_input": state.get("user_input", ""),
                "input_summary": state.get("input_summary", ""),
                "session_state": session_state,
            },
        )
        return {"reply": extract_reply(output), "session_state": session_state, "status": "processed"}

    def load_session_state(self, state: AgentState) -> dict[str, Any]:
        session_state = self._load_session_state(state)
        return {
            "session_state": session_state,
            "last_reply": session_state.get("last_reply") if session_state else None,
            "last_retrieval_query": session_state.get("last_retrieval_query") if session_state else None,
        }

    def check_last_reply(self, state: AgentState) -> dict[str, Any]:
        session_state = state.get("session_state") or {}
        current_user_id = state.get("current_user_id")
        last_active_user_id = session_state.get("last_active_user_id")
        if current_user_id and last_active_user_id and current_user_id != last_active_user_id:
            return {"last_reply": None, "error": "last_reply_belongs_to_another_user"}
        return {"last_reply": session_state.get("last_reply")}

    def reply_missing_context(self, state: AgentState) -> dict[str, Any]:
        return {
            "reply": {
                "should_reply": True,
                "content": "没有可修改的上一条回复，请先生成一条回复建议。",
                "reason": "missing_last_reply",
            },
            "status": "missing_context",
        }

    def revise_reply(self, state: AgentState) -> dict[str, Any]:
        session_state = state.get("session_state") or {}
        output = self.llm_client.generate_json(
            task="reply",
            inputs={
                "intent": "revise_reply",
                "input_summary": state.get("input_summary", ""),
                "last_reply": session_state.get("last_reply"),
                "last_analysis": session_state.get("last_analysis"),
                "last_chat_context": session_state.get("last_chat_context"),
                "last_intent": session_state.get("last_intent"),
            },
        )
        return {"reply": extract_reply(output), "status": "processed"}

    def check_user_context(self, state: AgentState) -> dict[str, Any]:
        current_user_id = state.get("current_user_id")
        if not current_user_id:
            return {"status": "missing_user_id"}

        if self.active_memory_cache.user_id and self.active_memory_cache.user_id != current_user_id:
            context_switch_sync_result = None
            dirty_memories = [dict(memory) for memory in self.active_memory_cache.get_dirty_memories()]
            if dirty_memories:
                context_switch_sync_result = self._sync_dirty_memory_entries(dirty_memories)
                if context_switch_sync_result["sync_errors"]:
                    return {
                        "status": "sync_failed",
                        "error": self._merge_error(state.get("error"), "context_switch_sync_failed"),
                        "context_switch_sync_result": context_switch_sync_result,
                    }
                self.active_memory_cache.clear_dirty()
            self.active_memory_cache.clear()
        else:
            context_switch_sync_result = state.get("context_switch_sync_result")

        session_state = self.memory_store.get_session_state(state.get("me_id") or "default", current_user_id)
        working_memory = self.memory_store.get_working_memory_observations(current_user_id)
        return {
            "active_user_id": current_user_id,
            "session_state": session_state,
            "working_memory": working_memory,
            "last_retrieval_query": session_state.get("last_retrieval_query") if session_state else None,
            "context_switch_sync_result": context_switch_sync_result,
            "status": "user_context_ready",
        }

    def reply_missing_user_id(self, state: AgentState) -> dict[str, Any]:
        if state.get("status") == "sync_failed":
            return {
                "reply": {
                    "should_reply": False,
                    "content": "",
                    "reason": "context_switch_sync_failed",
                },
                "status": "sync_failed",
            }
        return {
            "reply": {
                "should_reply": True,
                "content": "请先选择一个聊天对象，再进行回复建议或画像更新。",
                "reason": "missing_user_id",
            },
            "status": "missing_user_id",
        }

    def take_screenshot_or_mock(self, state: AgentState) -> dict[str, Any]:
        if state.get("chat_context"):
            return {}
        return {"screenshot_path": state.get("screenshot_path") or "mock://screenshot"}

    def ocr_vision_llm(self, state: AgentState) -> dict[str, Any]:
        existing_chat_context = state.get("chat_context") or {}
        output = self.llm_client.generate_json(
            task="ocr",
            inputs={
                "user_input": state.get("user_input", ""),
                "chat_context": existing_chat_context,
                "screenshot_base64": state.get("screenshot_base64"),
                "screenshot_path": state.get("screenshot_path"),
            },
        )
        llm_chat_context = extract_chat_context(output)
        chat_context = existing_chat_context or llm_chat_context
        return {
            "chat_context": chat_context,
            "chat_text": messages_to_chat_text(chat_context.get("recent_messages", [])),
            "working_memory_observations": extract_working_memory_observations(output),
        }

    def input_filter_node(self, state: AgentState) -> dict[str, Any]:
        result = self.input_filter.check({"chat_context": state.get("chat_context") or {}})
        if not result.should_process:
            return {"status": "skipped", "error": result.reason}
        return {"status": "input_ready"}

    def reply_input_skipped(self, state: AgentState) -> dict[str, Any]:
        return {
            "reply": {"should_reply": False, "content": "", "reason": state.get("error") or "input_skipped"},
            "memory_saved": False,
        }

    def update_working_memory(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("active_user_id") or state.get("current_user_id")
        if not user_id:
            return {}
        working_memory = self.memory_store.update_working_memory_observations(
            user_id,
            state.get("working_memory_observations") or [],
        )
        return {"working_memory": working_memory}

    def retrieval_query_llm(self, state: AgentState) -> dict[str, Any]:
        fallback_query = state.get("input_summary") or state.get("chat_text") or state.get("user_input", "")
        try:
            output = self.llm_client.generate_json(
                task="retrieval_query",
                inputs={
                    "intent": state.get("intent"),
                    "user_input": state.get("user_input", ""),
                    "input_summary": state.get("input_summary", ""),
                    "chat_context": state.get("chat_context", {}),
                    "chat_text": state.get("chat_text", ""),
                    "working_memory": state.get("working_memory", []),
                },
            )
            retrieval_query = extract_retrieval_query(output) or fallback_query
            return {"retrieval_query": retrieval_query}
        except Exception as exc:
            return {
                "retrieval_query": fallback_query,
                "error": self._merge_error(state.get("error"), f"retrieval_query_failed:{exc}"),
            }

    def query_similarity_check(self, state: AgentState) -> dict[str, Any]:
        current_embedding = self.semantic_retriever.embed_text(state.get("retrieval_query", ""))
        last_embedding = self.semantic_retriever.embed_text(state.get("last_retrieval_query") or "")
        similarity = self.semantic_retriever.cosine_similarity(current_embedding, last_embedding)
        reuse_cache = similarity >= 0.85 and self.active_memory_cache.has_cache(state.get("active_user_id"))
        return {"query_similarity": similarity, "reuse_cache": reuse_cache}

    def reuse_cache(self, state: AgentState) -> dict[str, Any]:
        if not self.active_memory_cache.has_cache(state.get("active_user_id")):
            return self.retrieve_and_build_cache(state)
        memories = self.active_memory_cache.get_memories()
        return {"active_memory_cache": self.active_memory_cache.to_dict(), "relevant_memories": memories}

    def retrieve_and_build_cache(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("active_user_id") or state.get("current_user_id")
        retrieval_query = state.get("retrieval_query", "")
        semantic_results = self.semantic_retriever.query(
            user_id=user_id or "",
            query_text=retrieval_query,
            top_k=5,
            statuses=["stable", "pending", "conflict"],
        )
        memory_ids = [int(result["memory_id"]) for result in semantic_results if result.get("memory_id") is not None]
        records = [
            record
            for record in self.memory_store.get_memory_records(memory_ids)
            if record.get("user_id") == user_id
        ]
        self.active_memory_cache.set_cache(
            user_id=user_id or "",
            retrieval_query=retrieval_query,
            memories=records,
            query_embedding=self.semantic_retriever.embed_text(retrieval_query),
        )
        return {
            "semantic_results": semantic_results,
            "relevant_memories": records,
            "active_memory_cache": self.active_memory_cache.to_dict(),
        }

    def reply_advice_llm(self, state: AgentState) -> dict[str, Any]:
        stable_memories = [
            memory
            for memory in self.active_memory_cache.get_memories(["stable"])
            if memory.get("user_id") == state.get("active_user_id")
        ]
        output = self.llm_client.generate_json(
            task="reply",
            inputs={
                "intent": "reply_advice",
                "chat_context": state.get("chat_context", {}),
                "input_summary": state.get("input_summary", ""),
                "working_memory": state.get("working_memory", []),
                "memories": stable_memories,
            },
        )
        return {"reply": extract_reply(output)}

    def learning_llm(self, state: AgentState) -> dict[str, Any]:
        try:
            output = self.llm_client.generate_json(
                task="learning",
                inputs={
                    "intent": state.get("intent"),
                    "chat_context": state.get("chat_context", {}),
                    "input_summary": state.get("input_summary", ""),
                    "memories": self.active_memory_cache.get_memories(),
                },
            )
            return {
                "memory_updates": extract_memory_updates(output),
                "memory_reviews": extract_memory_reviews(output),
                "changed_summary": extract_changed_summary(output),
            }
        except Exception as exc:
            return {
                "memory_updates": [],
                "memory_reviews": [],
                "changed_summary": "",
                "error": self._merge_error(state.get("error"), f"learning_failed:{exc}"),
            }

    def update_cache_from_learning(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("active_user_id") or state.get("current_user_id")
        for memory_update in state.get("memory_updates", []):
            content = str(memory_update.get("content", "")).strip()
            if not content:
                continue
            confidence = float(memory_update.get("confidence", 0.8))
            has_conflict = memory_update.get("has_conflict") is True
            memory_status = classify_memory_status(confidence, has_conflict)
            if memory_status == "discard":
                continue
            self.active_memory_cache.add_pending_memory(
                {
                    "user_id": user_id,
                    "memory_type": memory_update.get("memory_type"),
                    "content": content,
                    "confidence": confidence,
                    "memory_status": memory_status,
                    "source_type": state.get("intent") or "unknown",
                    "source_summary": state.get("changed_summary") or state.get("input_summary"),
                    "last_evidence": memory_update.get("evidence"),
                }
            )

        reviewed_memories = self._apply_memory_reviews_to_cache(state.get("memory_reviews", []))
        return {
            "saved_memory_ids": [],
            "reviewed_memories": reviewed_memories,
            "dirty_memories": self.active_memory_cache.get_dirty_memories(),
            "active_memory_cache": self.active_memory_cache.to_dict(),
        }

    def sync_dirty_memory(self, state: AgentState) -> dict[str, Any]:
        dirty_memories = [dict(memory) for memory in self.active_memory_cache.get_dirty_memories()]
        sync_result = self._sync_dirty_memory_entries(dirty_memories)
        if not sync_result["sync_errors"]:
            self.active_memory_cache.clear_dirty()
        return {
            "dirty_memories": dirty_memories,
            "saved_memory_ids": sync_result["saved_memory_ids"],
            "reviewed_memories": sync_result["reviewed_memories"],
            "discarded_memory_ids": sync_result["discarded_memory_ids"],
            "sync_errors": sync_result["sync_errors"],
            "active_memory_cache": self.active_memory_cache.to_dict(),
            "error": self._merge_error(state.get("error"), "sync_failed") if sync_result["sync_errors"] else state.get("error"),
        }

    def profile_update_confirm_reply(self, state: AgentState) -> dict[str, Any]:
        output = self.llm_client.generate_json(
            task="reply",
            inputs={
                "intent": "profile_update",
                "input_summary": state.get("input_summary", ""),
                "changed_summary": state.get("changed_summary"),
            },
        )
        return {"reply": extract_reply(output)}

    def save_session_state(self, state: AgentState) -> dict[str, Any]:
        if state.get("status") in {"skipped", "missing_user_id", "missing_context", "sync_failed"}:
            return {
                "session_state_saved": False,
                "reason": f"not_saved_for_status:{state.get('status')}",
            }

        try:
            me_id = state.get("me_id") or "default"
            user_id = self._session_user_id(state)
            data = {
                "last_intent": state.get("intent"),
                "last_user_input": state.get("user_input"),
                "last_input_summary": state.get("input_summary"),
                "last_reply": state.get("reply"),
                "last_analysis": (state.get("reply") or {}).get("analysis"),
                "last_chat_context": state.get("chat_context"),
                "last_active_user_id": state.get("active_user_id") or state.get("current_user_id"),
                "last_retrieval_query": state.get("retrieval_query"),
            }
            self.memory_store.save_session_state(me_id, user_id, data)
            return {"session_state_saved": True}
        except Exception as exc:
            return {
                "session_state_saved": False,
                "error": self._merge_error(state.get("error"), f"session_save_failed:{exc}"),
            }

    def noop(self, state: AgentState) -> dict[str, Any]:
        return {}

    def mark_task_done(self, state: AgentState) -> dict[str, Any]:
        current_task = state.get("current_task") or state.get("intent")
        if not current_task:
            return {}
        completed_tasks = list(state.get("completed_tasks", []))
        if current_task not in completed_tasks:
            completed_tasks.append(current_task)

        task_results = list(state.get("task_results", []))
        task_results.append(
            {
                "task": current_task,
                "status": state.get("status"),
                "reply": state.get("reply"),
                "saved_memory_ids": state.get("saved_memory_ids", []),
                "reviewed_memories": state.get("reviewed_memories", []),
                "discarded_memory_ids": state.get("discarded_memory_ids", []),
                "session_state_saved": state.get("session_state_saved"),
                "error": state.get("error"),
            }
        )
        return {
            "completed_tasks": completed_tasks,
            "task_results": task_results,
        }

    def route_by_intent(self, state: AgentState) -> str:
        if len(state.get("completed_tasks", [])) >= len(state.get("task_list", [])):
            return "done"
        intent = state.get("intent") or "general_question"
        return intent if intent in ALLOWED_INTENTS else "general_question"

    def route_last_reply(self, state: AgentState) -> str:
        return "has_last_reply" if state.get("last_reply") else "no_last_reply"

    def route_after_user_context(self, state: AgentState) -> str:
        if state.get("status") in {"missing_user_id", "sync_failed"}:
            return "missing_user_id"
        return state.get("intent") or "reply_advice"

    def route_input_filter(self, state: AgentState) -> str:
        return "skipped" if state.get("status") == "skipped" else "ok"

    def route_cache(self, state: AgentState) -> str:
        return "reuse_cache" if state.get("reuse_cache") else "retrieve_and_build_cache"

    def route_after_cache(self, state: AgentState) -> str:
        return "profile_update" if state.get("intent") == "profile_update" else "reply_advice"

    def route_after_sync(self, state: AgentState) -> str:
        return "profile_update" if state.get("intent") == "profile_update" else "reply_advice"

    def _apply_memory_reviews_to_cache(self, memory_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reviewed: list[dict[str, Any]] = []
        for memory_review in memory_reviews:
            if "memory_id" not in memory_review or "confidence" not in memory_review:
                continue
            memory_id = int(memory_review["memory_id"])
            confidence = float(memory_review["confidence"])
            has_conflict = memory_review.get("has_conflict") is True
            memory_status = classify_memory_status(confidence, has_conflict)
            record = self._get_cached_memory(memory_id) or self.memory_store.get_memory_record(memory_id)
            if not record:
                reviewed.append(
                    {
                        "memory_id": memory_id,
                        "reviewed": False,
                        "confidence": confidence,
                        "has_conflict": has_conflict,
                        "memory_status": memory_status,
                    }
                )
                continue

            updated_record = dict(record)
            updated_record["confidence"] = confidence
            updated_record["memory_status"] = memory_status
            self.active_memory_cache.upsert_memory(updated_record, dirty=True)
            reviewed.append(
                {
                    "memory_id": memory_id,
                    "reviewed": True,
                    "confidence": confidence,
                    "has_conflict": has_conflict,
                    "memory_status": memory_status,
                }
            )
        return reviewed

    def _sync_dirty_memory_entries(self, dirty_memories: list[dict[str, Any]]) -> dict[str, Any]:
        saved_memory_ids: list[int] = []
        reviewed_memories: list[dict[str, Any]] = []
        discarded_memory_ids: list[int] = []
        sync_errors: list[dict[str, Any]] = []
        for memory in dirty_memories:
            memory_id = memory.get("id")
            memory_status = memory.get("memory_status")
            confidence = float(memory.get("confidence", 0.8))
            is_new_memory = memory.get("_is_new") is True or str(memory_id).startswith("tmp_")

            try:
                if is_new_memory:
                    if memory_status == "discard":
                        self.active_memory_cache.remove_memory(memory_id)
                        continue
                    real_id = self.memory_store.save_memory(
                        user_id=memory["user_id"],
                        memory_type=memory.get("memory_type"),
                        content=memory["content"],
                        confidence=confidence,
                        memory_status=memory_status,
                        source_type=memory.get("source_type") or "unknown",
                        source_summary=memory.get("source_summary"),
                        last_evidence=memory.get("last_evidence"),
                    )
                    saved_memory_ids.append(real_id)
                    real_record = self.memory_store.get_memory_record(real_id)
                    if real_record:
                        self.active_memory_cache.replace_memory_id(str(memory_id), real_id, real_record)
                    try:
                        self.semantic_retriever.add_memory(
                            memory_id=real_id,
                            user_id=memory["user_id"],
                            content=memory["content"],
                            memory_status=memory_status,
                            memory_type=memory.get("memory_type"),
                        )
                    except Exception as exc:
                        sync_errors.append(
                            {
                                "memory_id": real_id,
                                "error": f"index_sync_failed:{exc}",
                            }
                        )
                        self.active_memory_cache.mark_dirty(real_id)
                    continue

                real_memory_id = int(memory_id)
                if memory.get("memory_status") == "discard":
                    reviewed = self.memory_store.update_memory_review(
                        real_memory_id,
                        confidence,
                        "discard",
                    )
                    if not reviewed:
                        sync_errors.append(
                            {
                                "memory_id": real_memory_id,
                                "error": "memory_review_update_failed",
                            }
                        )
                        continue
                    try:
                        self.semantic_retriever.delete_memory(real_memory_id)
                    except Exception as exc:
                        sync_errors.append(
                            {
                                "memory_id": real_memory_id,
                                "error": f"index_sync_failed:{exc}",
                            }
                        )
                        continue
                    self.active_memory_cache.remove_memory(real_memory_id)
                    discarded_memory_ids.append(real_memory_id)
                    reviewed_memories.append(
                        {
                            "memory_id": real_memory_id,
                            "reviewed": True,
                            "confidence": confidence,
                            "memory_status": "discard",
                        }
                    )
                    continue

                reviewed = self.memory_store.update_memory_review(
                    real_memory_id,
                    confidence,
                    memory_status,
                )
                if not reviewed:
                    sync_errors.append(
                        {
                            "memory_id": real_memory_id,
                            "error": "memory_review_update_failed",
                        }
                    )
                    continue
                record = self.memory_store.get_memory_record(real_memory_id)
                if record:
                    try:
                        self.semantic_retriever.add_memory(
                            memory_id=real_memory_id,
                            user_id=record["user_id"],
                            content=record["content"],
                            memory_status=record["memory_status"],
                            memory_type=record.get("memory_type"),
                        )
                    except Exception as exc:
                        sync_errors.append(
                            {
                                "memory_id": real_memory_id,
                                "error": f"index_sync_failed:{exc}",
                            }
                        )
                        continue
                    self.active_memory_cache.upsert_memory(record, dirty=False)
                reviewed_memories.append(
                    {
                        "memory_id": real_memory_id,
                        "reviewed": True,
                        "confidence": confidence,
                        "memory_status": memory_status,
                    }
                )
            except Exception as exc:
                sync_errors.append(
                    {
                        "memory_id": memory_id,
                        "error": f"memory_sync_failed:{exc}",
                    }
                )
        return {
            "dirty_memories": dirty_memories,
            "saved_memory_ids": saved_memory_ids,
            "reviewed_memories": reviewed_memories,
            "discarded_memory_ids": discarded_memory_ids,
            "sync_errors": sync_errors,
            "active_memory_cache": self.active_memory_cache.to_dict(),
        }

    def _get_cached_memory(self, memory_id: int) -> dict[str, Any] | None:
        for memory in self.active_memory_cache.memories:
            if memory.get("id") == memory_id:
                return dict(memory)
        return None

    def _merge_error(self, current_error: str | None, new_error: str) -> str:
        if not current_error:
            return new_error
        if new_error in current_error:
            return current_error
        return f"{current_error}; {new_error}"

    def _load_session_state(self, state: AgentState) -> dict[str, Any] | None:
        me_id = state.get("me_id") or "default"
        current_user_id = state.get("current_user_id")
        if current_user_id:
            session_state = self.memory_store.get_session_state(me_id, current_user_id)
            if session_state:
                return session_state
        return self.memory_store.get_session_state(me_id, "global")

    def _session_user_id(self, state: AgentState) -> str:
        return state.get("current_user_id") or "global"

    def _initial_state(self, payload: dict[str, Any]) -> AgentState:
        chat_context = dict(payload.get("chat_context") or {})
        if payload.get("previous_recent_messages") is not None and "previous_recent_messages" not in chat_context:
            chat_context["previous_recent_messages"] = payload["previous_recent_messages"]
        return {
            "me_id": payload.get("me_id") or "default",
            "current_user_id": payload.get("current_user_id"),
            "active_user_id": payload.get("current_user_id"),
            "user_input": payload.get("user_input", ""),
            "task_list": [],
            "current_task": None,
            "completed_tasks": [],
            "task_results": [],
            "chat_context": chat_context,
            "chat_text": messages_to_chat_text(chat_context.get("recent_messages", [])),
            "screenshot_path": payload.get("screenshot_path"),
            "screenshot_base64": payload.get("screenshot_base64"),
            "working_memory_observations": payload.get("working_memory_observations", []),
            "working_memory": [],
            "semantic_results": [],
            "relevant_memories": [],
            "memory_updates": [],
            "memory_reviews": [],
            "dirty_memories": [],
            "saved_memory_ids": [],
            "reviewed_memories": [],
            "discarded_memory_ids": [],
            "sync_errors": [],
            "context_switch_sync_result": None,
            "status": "started",
            "error": None,
        }

    def _normalize_task_list(self, raw_intents: Any) -> list[str]:
        if not isinstance(raw_intents, list):
            raw_intents = [raw_intents]
        seen: set[str] = set()
        allowed = {intent for intent in raw_intents if isinstance(intent, str) and intent in ALLOWED_INTENTS}
        ordered: list[str] = []
        for intent in TASK_PRIORITY:
            if intent in allowed and intent not in seen:
                ordered.append(intent)
                seen.add(intent)
        return ordered or ["general_question"]

    def _public_result(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": state.get("status"),
            "intent": state.get("intent"),
            "task_list": state.get("task_list", []),
            "current_task": state.get("current_task"),
            "completed_tasks": state.get("completed_tasks", []),
            "task_results": state.get("task_results", []),
            "input_summary": state.get("input_summary"),
            "active_user_id": state.get("active_user_id"),
            "reply": state.get("reply"),
            "working_memory": state.get("working_memory", []),
            "retrieval_query": state.get("retrieval_query"),
            "query_similarity": state.get("query_similarity"),
            "reuse_cache": state.get("reuse_cache"),
            "saved_memory_ids": state.get("saved_memory_ids", []),
            "reviewed_memories": state.get("reviewed_memories", []),
            "discarded_memory_ids": state.get("discarded_memory_ids", []),
            "session_state_saved": state.get("session_state_saved"),
            "error": state.get("error"),
            "debug": {
                "semantic_results": state.get("semantic_results", []),
                "relevant_memories": state.get("relevant_memories", []),
                "dirty_memories": state.get("dirty_memories", []),
                "sync_errors": state.get("sync_errors", []),
                "context_switch_sync_result": state.get("context_switch_sync_result"),
            },
        }