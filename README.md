# Context-Aware Memory Agent

This project is a Python backend for a long-term memory conversation agent.

Current main stack:

```text
chat input
-> InputFilter
-> LangGraph state graph
-> Generic LLMClient tasks: intent / reply / ocr / retrieval_query / learning
-> SQLite source of truth
-> SemanticRetriever / ChromaDB recall index
-> ActiveMemoryCache
-> session_state + working_memory_observations
```

The main chain does not depend on any fixed model platform. `LLMClient` is an OpenAI-compatible JSON client for the runtime path. `MockLLMClient` is kept for tests and explicit local doubles only.

## Project Structure

```text
memory_agent/
  graph_agent.py          # LangGraph GraphMemoryAgent main entry
  llm_client.py           # Generic BaseLLMClient / MockLLMClient / LLMClient
  state.py                # AgentState
  active_memory_cache.py  # Active user memory cache
  memory_store.py         # SQLite long-term/session/working memory storage
  semantic_retriever.py   # ChromaDB semantic recall index with fallback
  models.py               # Generic model output parsing helpers
  input_filter.py         # Empty/duplicate chat filtering
  agent.py                # Legacy sequential agent
examples/
  sample_input.json
tests/
  test_graph_agent_local.py
  test_memory_store.py
  test_semantic_retriever.py
  test_models.py
main.py
```

## Environment

```text
MEMORY_DB_PATH=memory.db
CHROMA_DB_PATH=chroma_memory
CHROMA_COLLECTION_NAME=user_memory_bge_base_zh_v15
EMBEDDING_MODEL_NAME=BAAI/bge-base-zh-v1.5
EMBEDDING_QUERY_INSTRUCTION=为这个句子生成表示以用于检索相关文章：
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.2
LLM_RESPONSE_FORMAT=json_object
```

## Semantic Retrieval

`SemanticRetriever` uses `BAAI/bge-base-zh-v1.5` by default for Chinese memory
retrieval. Query text is embedded with the BGE query instruction configured by
`EMBEDDING_QUERY_INSTRUCTION`; stored memory content is embedded as-is.

SQLite remains the source of truth. ChromaDB is only the recall index. The
default collection name is model-specific (`user_memory_bge_base_zh_v15`) to
avoid mixing vectors from different embedding dimensions in one collection.

## Run

```bash
python main.py examples/sample_input.json
```

## Dependencies

Core SQLite behavior uses the Python standard library. Optional graph and semantic retrieval dependencies:

```text
langgraph
chromadb
sentence-transformers
```

If `chromadb` / `sentence-transformers` are not installed or cannot load locally, `SemanticRetriever` falls back to a simple lexical index. If `langgraph` is not installed, `graph_agent.py` includes a minimal local runner for development tests.

## Test

```bash
python -m unittest discover -s tests
```
