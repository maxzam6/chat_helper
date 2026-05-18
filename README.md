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

The main chain does not depend on any fixed model platform. `LLMClient` is a thin abstraction that can later be connected to OpenAI, local models, Ollama, Tongyi, or other providers. `MockLLMClient` keeps the local loop runnable without real keys.

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
```

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
