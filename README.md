# Efficient Text-to-SQL — DuckDB Analytics Agent

A **multi-agent NL-to-SQL system** built with LangGraph, DuckDB, and OpenAI (or Ollama).  
Zero LLM code execution — the LLM only produces SQL strings; all computation is deterministic.

## What's New — Multi-Table Support

The agent now handles multiple tables simultaneously with automatic schema discovery and join inference.

### Schema Catalog
Every ingested table is indexed into a persistent `_column_catalog` inside DuckDB. Each column record stores its type, semantic role (`is_metric`, `is_dimension`, `is_join_key`), and sample values. The catalog survives connection restarts via `_auto_reattach_parquet`.

### Relationship Registry
`_relationships` stores authoritative join keys between tables. On every `load_file` call, the agent auto-detects probable join columns by name and type matching. Relationships can also be registered manually via `register_relationship_tool`.

### New Tools
| Tool | Purpose |
|---|---|
| `load_file` | Load Excel / CSV / Parquet → Parquet on disk → DuckDB VIEW + catalog |
| `search_semantic_lookup` | Find columns across all tables by keyword or description |
| `get_schema_context_tool` | Full schema block for a set of tables injected into LLM prompts |
| `register_relationship_tool` | Explicitly record authoritative join keys |
| `get_relationships` | List all known cross-table relationships |
| `list_tables` | Summary of all loaded tables with row/column counts |

### Lazy Database Dispatch
All calls from `tools.py` into `agent.database` resolve through `sys.modules` at call-time rather than a module-level import. This eliminates stale-binding bugs in test environments where the database module is reset between tests.

---

---

## Current State

### What works today

- **Multi-agent graph** — Orchestrator → Profiler → SQL Writer → Execute SQL → Verifier pipeline,
  fully wired in LangGraph with `MemorySaver` for multi-turn conversation in Studio.
- **File ingestion** — `load_file` accepts `.xlsx`, `.xls`, `.csv`, and `.parquet`;
  Excel/CSV inputs are normalised to Parquet immediately on ingest.
- **Persistence** — when `DUCKDB_PATH` points to a file, all metadata and data survive restarts;
  `auto_reattach_parquet()` re-registers every previously loaded dataset as a DuckDB view on startup,
  so analysts never re-upload.
- **Schema catalog** — `_schema_catalog` stores column names, types, and sample values for every
  loaded dataset; used by the Profiler and injected into SQL-writer prompts.
- **Semantic lookup** — `_semantic_lookup` stores per-column statistics (distinct count, null
  fraction, top samples, inferred description) generated automatically at ingest time.
- **Business-term aliases** — `_semantic_map` maps natural-language terms to exact column names;
  fully queryable by the agent and editable by the user via `register_alias` or the generated
  CSV template.
- **`load_file_node` graph node** — the orchestrator now routes `load …` intents to
  `load_file_node` (previously the node existed but was never wired into the graph).
- **Pluggable LLM backend** — `LLM_BACKEND=openai` (default) or `LLM_BACKEND=ollama`;
  model and base URL configurable via env vars.
- **CLI wrapper** (`cli.py`) — backend and model selectable as CLI flags without touching `.env`.

### Architecture

```
User Query
    │
    ▼
┌─────────────┐   load?    ┌──────────────────┐
│ Orchestrator│────────────▶  load_file_node   │
│  (planner)  │            └──────────────────┘
│             │   plan     ┌──────────────┐
│             │────────────▶   set_step   │
│             │◀─── done ──│  (router)    │
└─────────────┘            └──────┬───────┘
        ▲                         │ profile? ──▶ Profiler
        │                         │ sql?     ──▶ SQL Writer
        │                                              │
        │                                       Execute SQL
        │                                              │
        └──────────────── orchestrator ◀─── Verifier ◀┘
```

**Nodes:**

| Node | Role |
|---|---|
| `orchestrator` | Decomposes NL query into a plan; synthesises the final answer |
| `load_file_node` | Ingests Excel / CSV / Parquet → Parquet + DuckDB view + registry entry |
| `profiler` | Explores column distributions to ground SQL semantics |
| `sql_writer` | Generates precise, read-only DuckDB SQL |
| `execute_sql` | Deterministic; runs the query, returns rows |
| `verifier` | Cross-checks results with test queries and checksums |

---

---

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Python ≥ 3.11
- OpenAI API key

### 1 — Install

```bash
uv sync --group dev
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env — minimum: set OPENAI_API_KEY
# For persistence also set DUCKDB_PATH and PARQUET_STORE (see below)
```

### 3 — Run

```bash
# OpenAI (default)
uv run cli.py

# Ollama — local, no API key needed
uv run cli.py --backend ollama
uv run cli.py --backend ollama --model gemma4:e4b
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

---

## Running Tests

```bash
# All unit tests (no LLM required)
pytest tests/ -v --ignore=tests/test_langgraph_query_rounds.py

# Single LLM round (LangGraph Studio must be running)
pytest tests/test_langgraph_query_rounds.py -m round_01 -v

# Resume from a specific round
pytest tests/test_langgraph_query_rounds.py -m "round_07 or round_08 or round_09 or round_10 or round_11 or round_12" -v

# Skip multi-table rounds
pytest tests/test_langgraph_query_rounds.py -m "not round_12" -v
```

---

---

## Tools Reference

| Tool | Description |
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
