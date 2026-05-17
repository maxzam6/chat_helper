# 基于长期记忆机制的上下文感知智能对话 Agent

当前阶段实现一个最小可运行的 Python Backend：

```text
聊天输入
-> InputFilter
-> SQLite working memory
-> Dify 生成 retrieval_query
-> ChromaDB 语义召回 relevant_memories
-> Dify 输出 reply / memory_updates / memory_reviews / updated_working_memory
-> SQLite + ChromaDB 更新
```

本项目不负责 Dify 工作流搭建，只提供后端对接边界。

## 项目结构

```text
memory_agent/
  agent.py          # Agent 编排主流程
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

如果没有配置 `DIFY_API_KEY`，程序会使用本地 mock 分析结果，方便先验证 SQLite 和 Agent 闭环。

## 依赖

基础 SQLite 流程只依赖 Python 标准库。语义检索使用：

```text
chromadb
sentence-transformers
```

如果本地暂时没有安装这些依赖，`SemanticRetriever` 会自动降级到轻量 fallback，保证后端流程还能跑通。

## 测试

```bash
python -m unittest discover -s tests
```
