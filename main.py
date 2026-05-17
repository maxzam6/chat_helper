from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from memory_agent.agent import MemoryAgent
from memory_agent.dify_client import DifyClient
from memory_agent.memory_store import MemoryStore
from memory_agent.semantic_retriever import SemanticRetriever


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python main.py <input-json-file>")
        return 2

    # 命令行入口：读取一个 JSON 文件作为聊天输入。
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    # MEMORY_DB_PATH 控制 SQLite 文件位置；默认就是项目根目录 memory.db。
    memory_store = MemoryStore(os.getenv("MEMORY_DB_PATH", "memory.db"))

    # 没有配置 DIFY_API_KEY 时会走本地 mock，方便先测试后端闭环。
    dify_client = DifyClient(
        api_key=os.getenv("DIFY_API_KEY"),
        api_base=os.getenv("DIFY_API_BASE", "https://api.dify.ai/v1"),
        user=os.getenv("DIFY_USER", "memory-agent"),
    )
    semantic_retriever = SemanticRetriever(
        persist_path=os.getenv("CHROMA_DB_PATH", "chroma_memory"),
    )
    agent = MemoryAgent(memory_store, dify_client, semantic_retriever=semantic_retriever)
    result = agent.process(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
