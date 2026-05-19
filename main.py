from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from memory_agent.graph_agent import GraphMemoryAgent
from memory_agent.llm_client import LLMClient
from memory_agent.memory_store import MemoryStore
from memory_agent.semantic_retriever import SemanticRetriever
from dotenv import load_dotenv

load_dotenv()

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python main.py <input-json-file>")
        return 2

    payload_path = Path(sys.argv[1])
    payload = normalize_payload(json.loads(payload_path.read_text(encoding="utf-8")))

    memory_store = MemoryStore(os.getenv("MEMORY_DB_PATH", "memory.db"))
    semantic_retriever = SemanticRetriever(
        persist_path=os.getenv("CHROMA_DB_PATH", "chroma_memory"),
    )
    agent = GraphMemoryAgent(
        memory_store=memory_store,
        llm_client=LLMClient(),
        semantic_retriever=semantic_retriever,
    )
    result = agent.process(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def normalize_payload(payload: dict) -> dict:
    """Keep older sample JSON usable with the graph agent entrypoint."""
    normalized = dict(payload)
    if "current_user_id" not in normalized and normalized.get("user_id"):
        normalized["current_user_id"] = normalized["user_id"]
    if "user_input" not in normalized and normalized.get("chat_context"):
        normalized["user_input"] = "这句怎么回？"
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
