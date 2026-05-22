from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .active_memory_cache import ActiveMemoryCache
from .graph_agent import GraphMemoryAgent
from .hotkey_capture import HotkeyCaptureService
from .llm_client import LLMClient
from .memory_store import MemoryStore
from .semantic_retriever import SemanticRetriever
from .vision_llm_client import VisionLLMClient

load_dotenv()

app = FastAPI(title="Chat Helper Memory Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    store = MemoryStore(os.getenv("MEMORY_DB_PATH", "memory.db"))
    store.init_db()
    return store


@lru_cache(maxsize=1)
def get_semantic_retriever() -> SemanticRetriever:
    return SemanticRetriever(
        persist_path=os.getenv("CHROMA_DB_PATH", "chroma_memory"),
        collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            SemanticRetriever.DEFAULT_COLLECTION_NAME,
        ),
        model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            SemanticRetriever.DEFAULT_MODEL_NAME,
        ),
        query_instruction=os.getenv(
            "EMBEDDING_QUERY_INSTRUCTION",
            SemanticRetriever.DEFAULT_QUERY_INSTRUCTION,
        ),
    )


@lru_cache(maxsize=1)
def get_agent() -> GraphMemoryAgent:
    return GraphMemoryAgent(
        memory_store=get_memory_store(),
        llm_client=LLMClient(),
        vision_llm_client=get_vision_llm_client(),
        semantic_retriever=get_semantic_retriever(),
        active_memory_cache=ActiveMemoryCache(),
    )


@lru_cache(maxsize=1)
def get_hotkey_capture_service() -> HotkeyCaptureService:
    service = HotkeyCaptureService(agent_factory=get_agent)
    service.start()
    return service


@lru_cache(maxsize=1)
def get_vision_llm_client() -> VisionLLMClient | None:
    if not os.getenv("VISION_API_KEY") or not os.getenv("VISION_MODEL"):
        return None
    return VisionLLMClient()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def start_hotkey_capture() -> None:
    get_hotkey_capture_service()


@app.get("/users/suggest")
def suggest_users(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    store = get_memory_store()
    return {
        "query": query,
        "suggestions": store.find_similar_user_ids(query, limit=limit),
    }


@app.get("/users/{user_id}/memories")
def list_user_memories(user_id: str) -> dict[str, Any]:
    store = get_memory_store()
    return {"user_id": user_id, "memories": store.get_user_memory_records(user_id)}


@app.get("/users/{user_id}/working-memory")
def list_working_memory(user_id: str) -> dict[str, Any]:
    store = get_memory_store()
    return {
        "user_id": user_id,
        "working_memory": store.get_working_memory_observations(user_id),
    }


@app.post("/agent/process")
def process_agent(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return get_agent().process(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/capture/status")
def capture_status() -> dict[str, Any]:
    return get_hotkey_capture_service().status()


@app.post("/capture/context")
def update_capture_context(payload: dict[str, Any]) -> dict[str, Any]:
    return get_hotkey_capture_service().update_context(payload)


@app.post("/capture/trigger")
def trigger_capture() -> dict[str, Any]:
    return get_hotkey_capture_service().trigger()


@app.post("/capture/cancel")
def cancel_capture() -> dict[str, Any]:
    return get_hotkey_capture_service().cancel()
