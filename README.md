# Efficient Text-to-SQL — DuckDB Analytics Agent

A **state-of-the-art multi-agent NL-to-SQL system** built with LangGraph, DuckDB, and OpenAI.
Zero LLM code execution — the LLM only produces SQL strings; all computation is deterministic.

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     plan      ┌──────────────┐
│ Orchestrator│──────────────▶│   set_step   │
│  (planner)  │◀──── done ────│  (router)    │
└─────────────┘               └──────┬───────┘
        ▲                            │ profile? ─────▶ Profiler
        │                            │ sql?     ─────▶ SQL Writer
        │                                                   │
        │                                            Execute SQL
        │                                                   │
        └──────────────── orchestrator ◀──── Verifier ◀────┘
```

**Agents:**
- **Orchestrator** — decomposes the NL query into a plan, synthesises the final answer.
- **Profiler** — explores column distributions to ground SQL semantics.
- **SQL Writer** — generates precise DuckDB SQL (read-only SELECT only).
- **Execute SQL** — deterministic node, no LLM, runs the query.
- **Verifier** — cross-checks results with test queries and checksums.

## Quick Start

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Python ≥ 3.11
- OpenAI API key

### 1. Install dependencies

```bash
uv sync --group dev
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run LangGraph Studio

```bash
uv run langgraph dev
```

Open the URL printed in the terminal (usually `http://localhost:2024`) to chat with the agent interactively in LangGraph Studio.

### 4. Load a dataset and start chatting

From a Python session or test script:

```python
from agent.tools import load_file
load_file.invoke({"path": "data/sales.csv", "dataset_name": "sales"})
```

Then ask in Studio:
> *"What are the top 5 products by total revenue in Q4?"*

## Project Structure

```
efficient-text-to-sql/
├── agent/
│   ├── __init__.py
│   ├── state.py        # AnalyticsState (Pydantic) + PlanStep
│   ├── database.py     # DuckDB singleton + _schema_catalog / _semantic_map
│   ├── tools.py        # 6 deterministic @tool functions
│   ├── prompts.py      # System prompts for each node
│   ├── nodes.py        # LangGraph node functions
│   └── graph.py        # StateGraph → exports `graph` for Studio
├── langgraph.json      # LangGraph Studio config
├── pyproject.toml      # uv project + dependencies
├── .env.example        # Environment variable template
└── README.md
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Zero LLM code execution** | Safety + auditability — LLM outputs SQL strings only |
| **DuckDB internal metadata** | No external Postgres; `_schema_catalog` + `_semantic_map` live inside DuckDB |
| **MemorySaver checkpointer** | Multi-turn conversation in Studio without Redis/Postgres |
| **uv** | Fastest Python dependency resolver; single `uv sync` installs everything |
| **gpt-4o** | Best instruction-following for structured JSON outputs |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4o` | Model name override |
| `DUCKDB_PATH` | ❌ | `:memory:` | Path to a persistent DuckDB file |
| `LANGSMITH_API_KEY` | ❌ | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | ❌ | — | Set to `true` to enable tracing |
