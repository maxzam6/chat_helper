from __future__ import annotations

from typing import Any, Callable

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - exercised only when langgraph is missing
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
from .dify_client import DifyClient
from .input_filter import InputFilter
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


ALLOWED_INTENTS = {"general_question", "revise_reply", "reply_advice", "profile_update"}


class GraphMemoryAgent:
    """LangGraph-based memory agent.

    This class keeps the same storage/retrieval/Dify boundaries as the legacy
    MemoryAgent, but moves orchestration into a StateGraph.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        dify_client: DifyClient,
        input_filter: InputFilter | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        active_memory_cache: ActiveMemoryCache | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.dify_client = dify_client
        self.input_filter = input_filter or InputFilter()
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        self.active_memory_cache = active_memory_cache or ActiveMemoryCache()
        self.app = self.build_graph().compile()

    def build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("classify_intent", self.classify_intent)
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

        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "general_question": "reply_general",
                "revise_reply": "load_session_state",
                "reply_advice": "check_user_context",
                "profile_update": "check_user_context",
            },
        )

        graph.add_edge("reply_general", "save_session_state")

        graph.add_edge("load_session_state", "check_last_reply")
        graph.add_conditional_edges(
            "check_last_reply",
            self.route_last_reply,
            {
                "has_last_reply": "revise_reply",
                "no_last_reply": "reply_missing_context",
            },
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
            {
                "skipped": "reply_input_skipped",
                "ok": "update_working_memory",
            },
        )
        graph.add_edge("reply_input_skipped", "save_session_state")
        graph.add_edge("update_working_memory", "retrieval_query_llm")

        graph.add_edge("retrieval_query_llm", "query_similarity_check")
        graph.add_conditional_edges(
            "query_similarity_check",
            self.route_cache,
            {
                "reuse_cache": "reuse_cache",
                "retrieve_and_build_cache": "retrieve_and_build_cache",
            },
        )
        graph.add_edge("reuse_cache", "after_cache")
        graph.add_edge("retrieve_and_build_cache", "after_cache")
        graph.add_conditional_edges(
            "after_cache",
            self.route_after_cache,
            {
                "reply_advice": "reply_advice_llm",
                "profile_update": "learning_llm",
            },
        )

        graph.add_edge("reply_advice_llm", "learning_llm")
        graph.add_edge("learning_llm", "update_cache_from_learning")
        graph.add_edge("update_cache_from_learning", "sync_dirty_memory")
        graph.add_conditional_edges(
            "sync_dirty_memory",
            self.route_after_sync,
            {
                "reply_advice": "save_session_state",
                "profile_update": "profile_update_confirm_reply",
            },
        )
        graph.add_edge("profile_update_confirm_reply", "save_session_state")
        graph.add_edge("save_session_state", END)
        return graph

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.memory_store.init_db()
        initial_state = self._initial_state(payload)
        return self.app.invoke(initial_state)

    def classify_intent(self, state: AgentState) -> dict[str, Any]:
        output = self.dify_client.run_workflow(
            {
                "stage": "intent_classifier",
                "user_input": state.get("user_input", ""),
                "me_id": state.get("me_id"),
                "current_user_id": state.get("current_user_id"),
            }
        )
        result = extract_intent_result(output)
        intent = result.get("intent") or "general_question"
        if intent not in ALLOWED_INTENTS:
            intent = "general_question"
        return {
            "intent": intent,
            "input_summary": result.get("input_summary") or state.get("user_input", ""),
            "status": "intent_classified",
        }

    def reply_general(self, state: AgentState) -> dict[str, Any]:
        session_state = self.memory_store.get_session_state(
            state.get("me_id") or "default",
            "global",
        )
        output = self.dify_client.run_workflow(
            {
                "stage": "reply",
                "intent": "general_question",
                "user_input": state.get("user_input", ""),
                "input_summary": state.get("input_summary", ""),
                "session_state": session_state,
            }
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
            return {
                "last_reply": None,
                "error": "last_reply_belongs_to_another_user",
            }
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
        output = self.dify_client.run_workflow(
            {
                "stage": "reply",
                "intent": "revise_reply",
                "input_summary": state.get("input_summary", ""),
                "last_reply": session_state.get("last_reply"),
                "last_analysis": session_state.get("last_analysis"),
                "last_chat_context": session_state.get("last_chat_context"),
                "last_intent": session_state.get("last_intent"),
            }
        )
        return {"reply": extract_reply(output), "status": "processed"}

    def check_user_context(self, state: AgentState) -> dict[str, Any]:
        current_user_id = state.get("current_user_id")
        if not current_user_id:
            return {"status": "missing_user_id"}

        if self.active_memory_cache.user_id and self.active_memory_cache.user_id != current_user_id:
            self._sync_dirty_memory_entries(self.active_memory_cache.get_dirty_memories())
            self.active_memory_cache.clear()

        session_state = self.memory_store.get_session_state(
            state.get("me_id") or "default",
            current_user_id,
        )
        working_memory = self.memory_store.get_working_memory_observations(current_user_id)
        return {
            "active_user_id": current_user_id,
            "session_state": session_state,
            "working_memory": working_memory,
            "last_retrieval_query": session_state.get("last_retrieval_query") if session_state else None,
            "status": "user_context_ready",
        }

    def reply_missing_user_id(self, state: AgentState) -> dict[str, Any]:
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
        if state.get("chat_context"):
            return {"chat_text": messages_to_chat_text(state.get("chat_context", {}).get("recent_messages", []))}

        output = self.dify_client.run_workflow(
            {
                "stage": "ocr",
                "user_input": state.get("user_input", ""),
                "screenshot_base64": state.get("screenshot_base64"),
                "screenshot_path": state.get("screenshot_path"),
            }
        )
        chat_context = extract_chat_context(output)
        observations = extract_working_memory_observations(output)
        return {
            "chat_context": chat_context,
            "chat_text": messages_to_chat_text(chat_context.get("recent_messages", [])),
            "working_memory_observations": observations,
        }

    def input_filter_node(self, state: AgentState) -> dict[str, Any]:
        payload = {"chat_context": state.get("chat_context") or {}}
        result = self.input_filter.check(payload)
        if not result.should_process:
            return {"status": "skipped", "error": result.reason}
        return {"status": "input_ready"}

    def reply_input_skipped(self, state: AgentState) -> dict[str, Any]:
        return {
            "reply": {
                "should_reply": False,
                "content": "",
                "reason": state.get("error") or "input_skipped",
            },
            "memory_saved": False,
        }

    def update_working_memory(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("active_user_id") or state.get("current_user_id")
        if not user_id:
            return {}
        observations = state.get("working_memory_observations") or []
        working_memory = self.memory_store.update_working_memory_observations(
            user_id,
            observations,
        )
        return {"working_memory": working_memory}

    def retrieval_query_llm(self, state: AgentState) -> dict[str, Any]:
        output = self.dify_client.run_workflow(
            {
                "stage": "retrieval_query",
                "intent": state.get("intent"),
                "user_input": state.get("user_input", ""),
                "input_summary": state.get("input_summary", ""),
                "chat_context": state.get("chat_context", {}),
                "chatText": state.get("chat_text", ""),
                "working_memory": state.get("working_memory", []),
            }
        )
        retrieval_query = extract_retrieval_query(output)
        return {"retrieval_query": retrieval_query}

    def query_similarity_check(self, state: AgentState) -> dict[str, Any]:
        retrieval_query = state.get("retrieval_query", "")
        last_query = state.get("last_retrieval_query")
        current_embedding = self.semantic_retriever.embed_text(retrieval_query)
        last_embedding = self.semantic_retriever.embed_text(last_query or "")
        similarity = self.semantic_retriever.cosine_similarity(current_embedding, last_embedding)
        reuse_cache = (
            similarity >= 0.85
            and self.active_memory_cache.has_cache(state.get("active_user_id"))
        )
        return {
            "query_similarity": similarity,
            "reuse_cache": reuse_cache,
        }

    def reuse_cache(self, state: AgentState) -> dict[str, Any]:
        if not self.active_memory_cache.has_cache(state.get("active_user_id")):
            return self.retrieve_and_build_cache(state)
        memories = self.active_memory_cache.get_memories()
        return {
            "active_memory_cache": self.active_memory_cache.to_dict(),
            "relevant_memories": memories,
        }

    def retrieve_and_build_cache(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("active_user_id") or state.get("current_user_id")
        retrieval_query = state.get("retrieval_query", "")
        semantic_results = self.semantic_retriever.query(
            user_id=user_id or "",
            query_text=retrieval_query,
            top_k=5,
            statuses=["stable", "pending", "conflict"],
        )
        memory_ids = [
            int(result["memory_id"])
            for result in semantic_results
            if result.get("memory_id") is not None
        ]
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
        output = self.dify_client.run_workflow(
            {
                "stage": "reply",
                "intent": "reply_advice",
                "chat_context": state.get("chat_context", {}),
                "input_summary": state.get("input_summary", ""),
                "working_memory": state.get("working_memory", []),
                "memories": stable_memories,
            }
        )
        return {"reply": extract_reply(output)}

    def learning_llm(self, state: AgentState) -> dict[str, Any]:
        output = self.dify_client.run_workflow(
            {
                "stage": "learning",
                "intent": state.get("intent"),
                "chat_context": state.get("chat_context", {}),
                "input_summary": state.get("input_summary", ""),
                "memories": self.active_memory_cache.get_memories(),
            }
        )
        return {
            "memory_updates": extract_memory_updates(output),
            "memory_reviews": extract_memory_reviews(output),
            "changed_summary": extract_changed_summary(output),
        }

    def update_cache_from_learning(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("active_user_id") or state.get("current_user_id")
        saved_memory_ids: list[int] = []
        for memory_update in state.get("memory_updates", []):
            memory_id = self._apply_memory_update(user_id or "", memory_update)
            if memory_id is None:
                continue
            saved_memory_ids.append(memory_id)
            record = self.memory_store.get_memory_record(memory_id)
            if record:
                self.active_memory_cache.upsert_memory(record, dirty=True)

        reviewed_memories = self._apply_memory_reviews(state.get("memory_reviews", []))
        return {
            "saved_memory_ids": saved_memory_ids,
            "reviewed_memories": reviewed_memories,
            "dirty_memories": self.active_memory_cache.get_dirty_memories(),
            "active_memory_cache": self.active_memory_cache.to_dict(),
        }

    def sync_dirty_memory(self, state: AgentState) -> dict[str, Any]:
        dirty_memories = self.active_memory_cache.get_dirty_memories()
        self._sync_dirty_memory_entries(dirty_memories)
        self.active_memory_cache.clear_dirty()
        return {
            "dirty_memories": dirty_memories,
            "active_memory_cache": self.active_memory_cache.to_dict(),
        }

    def profile_update_confirm_reply(self, state: AgentState) -> dict[str, Any]:
        output = self.dify_client.run_workflow(
            {
                "stage": "reply",
                "intent": "profile_update",
                "input_summary": state.get("input_summary", ""),
                "changed_summary": state.get("changed_summary"),
            }
        )
        return {"reply": extract_reply(output)}

    def save_session_state(self, state: AgentState) -> dict[str, Any]:
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

    def noop(self, state: AgentState) -> dict[str, Any]:
        return {}

    def route_by_intent(self, state: AgentState) -> str:
        intent = state.get("intent") or "general_question"
        return intent if intent in ALLOWED_INTENTS else "general_question"

    def route_last_reply(self, state: AgentState) -> str:
        return "has_last_reply" if state.get("last_reply") else "no_last_reply"

    def route_after_user_context(self, state: AgentState) -> str:
        if state.get("status") == "missing_user_id":
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

    def _apply_memory_update(self, user_id: str, memory_update: dict[str, Any]) -> int | None:
        content = str(memory_update.get("content", "")).strip()
        if not content:
            return None
        confidence = float(memory_update.get("confidence", 0.8))
        has_conflict = memory_update.get("has_conflict") is True
        memory_status = classify_memory_status(confidence, has_conflict)
        if memory_status == "discard":
            return None
        return self.memory_store.save_memory(
            user_id=user_id,
            memory_type=memory_update.get("memory_type"),
            content=content,
            confidence=confidence,
            memory_status=memory_status,
        )

    def _apply_memory_reviews(self, memory_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reviewed: list[dict[str, Any]] = []
        for memory_review in memory_reviews:
            if "memory_id" not in memory_review or "confidence" not in memory_review:
                continue
            memory_id = int(memory_review["memory_id"])
            confidence = float(memory_review["confidence"])
            has_conflict = memory_review.get("has_conflict") is True
            new_status = self.memory_store.review_memory_status(memory_id, confidence, has_conflict)
            record = self.memory_store.get_memory_record(memory_id)
            if record:
                self.active_memory_cache.upsert_memory(record, dirty=True)
            reviewed.append(
                {
                    "memory_id": memory_id,
                    "reviewed": new_status is not None,
                    "confidence": confidence,
                    "has_conflict": has_conflict,
                    "memory_status": new_status,
                }
            )
        return reviewed

    def _sync_dirty_memory_entries(self, dirty_memories: list[dict[str, Any]]) -> None:
        for memory in dirty_memories:
            memory_id = int(memory["id"])
            if memory.get("memory_status") == "discard":
                self.semantic_retriever.delete_memory(memory_id)
                continue
            self.semantic_retriever.add_memory(
                memory_id=memory_id,
                user_id=memory["user_id"],
                content=memory["content"],
                memory_status=memory["memory_status"],
                memory_type=memory.get("memory_type"),
            )

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
            "status": "started",
            "error": None,
        }
