# 基于长期记忆机制的上下文感知智能对话 Agent

当前阶段实现一个最小可运行的 Python Backend：

```text
聊天输入
-> InputFilter
-> SQLite working memory
-> LangGraph 状态图编排
-> Dify 意图分类 / retrieval_query / reply / learning
-> ChromaDB 语义召回 relevant_memories
-> Dify 输出 reply / memory_updates / memory_reviews / updated_working_memory
-> SQLite + ChromaDB 更新
```

本项目不负责 Dify 工作流搭建，只提供后端对接边界。

## 项目结构

```text
memory_agent/
  agent.py          # Agent 编排主流程
  graph_agent.py    # LangGraph 状态图 Agent 入口
  state.py          # LangGraph AgentState
  active_memory_cache.py # 当前用户相关记忆缓存
  dify_client.py    # Dify Workflow HTTP 客户端
  input_filter.py   # 空聊天、重复聊天过滤
  memory_store.py   # SQLite 长期记忆系统
  semantic_retriever.py # ChromaDB 语义检索索引
  models.py         # 输入和输出数据结构辅助
examples/
  sample_input.json # 示例聊天输入
tests/
  test_agent.py
  test_input_filter.py
  test_memory_store.py
main.py             # 命令行运行入口
```

## 环境变量

复制 `.env.example` 后按你的 Dify 配置填写：

```text
DIFY_API_BASE=https://api.dify.ai/v1
DIFY_API_KEY=your-api-key
DIFY_USER=memory-agent
MEMORY_DB_PATH=memory.db
CHROMA_DB_PATH=chroma_memory
```

## 运行

```bash
python main.py examples/sample_input.json
```

如果没有配置 `DIFY_API_KEY`，程序会使用本地 mock 输出，方便先验证 SQLite、LangGraph、语义召回和 memory 写回闭环。

## 依赖

基础 SQLite 流程只依赖 Python 标准库。语义检索使用：

```text
chromadb
langgraph
sentence-transformers
```

如果本地暂时没有安装 `chromadb` / `sentence-transformers`，`SemanticRetriever` 会自动降级到轻量 fallback，保证后端流程还能跑通。

如果本地暂时没有安装 `langgraph`，`graph_agent.py` 内置了一个最小 fallback runner 用于本地测试；正式环境建议安装 `langgraph`。

## 测试

```bash
python -m unittest discover -s tests
```
