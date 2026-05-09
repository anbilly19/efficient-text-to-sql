# Efficient Text-to-SQL — DuckDB Analytics Agent

A **multi-agent NL-to-SQL system** built with LangGraph, DuckDB, and OpenAI (or Ollama).
Zero LLM code execution — the LLM only produces SQL strings; all computation is deterministic.

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     load?     ┌──────────────────┐
│ Orchestrator│──────────────▶│  load_file_node  │
│  (planner)  │               └──────────────────┘
│             │     plan      ┌──────────────┐
│             │──────────────▶│   set_step   │
│             │◀──── done ────│  (router)    │
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
- **load_file_node** — ingests Excel / CSV / Parquet, persists to Parquet, registers view.
- **Profiler** — explores column distributions to ground SQL semantics.
- **SQL Writer** — generates precise DuckDB SQL (read-only SELECT only).
- **Execute SQL** — deterministic node, no LLM, runs the query.
- **Verifier** — cross-checks results with test queries and checksums.

## Quick Start

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Python ≥ 3.11
- OpenAI API key **or** [Ollama](https://ollama.com) running locally

### 1. Install dependencies

```bash
uv sync --group dev
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY (or switch to Ollama, see below)
```

### 3. Run the agent

```bash
# OpenAI (default)
uv run cli.py

# Ollama — local, no API key needed
uv run cli.py --backend ollama
uv run cli.py --backend ollama --model gemma4:e4b

# Pass extra langgraph dev flags
uv run cli.py --backend ollama -- --port 8080
```

Open the URL printed in the terminal (usually `http://localhost:2024`) to chat in LangGraph Studio.

### 4. Load a dataset and start chatting

In the Studio chat:
```
load file at data/sales.xlsx as sales
```

Or from Python:
```python
from agent.tools import load_file
load_file.invoke({"path": "data/sales.xlsx", "dataset_name": "sales"})
```

Then ask:
> *"What are the top 5 products by total revenue in Q4?"*

## Persistence

By default the agent uses an **in-memory DuckDB** — all data is lost when the server stops.
To persist across restarts set these two env vars (or pass via `.env`):

```env
DUCKDB_PATH=./data/niq.duckdb   # file-backed database
PARQUET_STORE=./data/parquet    # where ingested files are stored as Parquet
```

On next startup the agent:
1. Opens the same DuckDB file (metadata tables intact)
2. Re-attaches all Parquet views via `_data_registry`
3. Makes every previously loaded dataset immediately queryable — no re-upload needed

## Business Term Aliases

When a file is loaded the agent checks whether any column has a business-term alias
registered in `_semantic_map`. If none exist, a CSV template is written to
`data/templates/<dataset>_aliases.csv`.

Fill in the `definition_sql` column (e.g. `"Umsatz_EUR"` or `SUM("Umsatz_EUR")/1000`)
and register it directly from the Studio chat:

```
register alias umsatz = "Umsatz_EUR" — net revenue in EUR
```

Or via tool:
```python
from agent.tools import register_alias
register_alias.invoke({
    "term": "umsatz",
    "definition_sql": '"Umsatz_EUR"',
    "description": "Net revenue in EUR",
})
```

Aliases are persisted in `_semantic_map` and used automatically by the SQL Writer.

## LLM Backends

| Backend | Command | Notes |
|---|---|---|
| OpenAI (default) | `uv run cli.py` | Requires `OPENAI_API_KEY` |
| OpenAI custom model | `uv run cli.py --model gpt-4o` | |
| Ollama local | `uv run cli.py --backend ollama` | Default model: `gemma4:e2b` |
| Ollama custom model | `uv run cli.py --backend ollama --model gemma4:e4b` | |
| Ollama custom URL | `uv run cli.py --backend ollama --ollama-url http://host:11434` | |

Install Ollama: https://ollama.com/download — then `ollama pull gemma4:e2b`.

## Project Structure

```
efficient-text-to-sql/
├── agent/
│   ├── __init__.py
│   ├── state.py        # AnalyticsState (Pydantic) + PlanStep
│   ├── database.py     # DuckDB singleton, metadata tables, Parquet restore
│   ├── tools.py        # Deterministic @tool functions
│   ├── prompts.py      # System prompts for each node
│   ├── nodes.py        # LangGraph node functions + LLM factory
│   └── graph.py        # StateGraph → exports `graph` for Studio
├── cli.py              # CLI wrapper — LLM backend as command-line arg
├── langgraph.json      # LangGraph Studio config
├── pyproject.toml      # uv project + dependencies
├── .env.example        # Environment variable template
└── README.md
```

## Tools Reference

| Tool | Description |
|---|---|
| `load_file` | Ingest Excel / CSV / Parquet → Parquet + DuckDB view |
| `list_loaded_tables` | Show all registered datasets (name, rows, source file) |
| `get_schema` | Column names, types, sample values for a dataset |
| `profile_column` | Null %, top-10 values, min/max/mean for a column |
| `run_sql` | Execute a read-only SELECT (first 50 rows) |
| `run_test_query` | Same as run_sql with checksums for verification |
| `lookup_semantic` | Resolve a business term to a SQL expression |
| `search_semantic_lookup` | Keyword search across column names and descriptions |
| `register_alias` | Add / update a business-term alias in `_semantic_map` |

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Zero LLM code execution** | Safety + auditability — LLM outputs SQL strings only |
| **Parquet-backed views** | Minimal memory; datasets survive restarts without re-upload |
| **`_data_registry`** | Tracks every ingested file so views can be re-attached on cold start |
| **`_semantic_map`** | Business-term aliases decouple German column names from English queries |
| **Pluggable LLM backend** | Switch OpenAI ↔ Ollama with one CLI flag, no code changes |
| **uv** | Fastest Python dependency resolver; single `uv sync` installs everything |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ (OpenAI) | — | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | OpenAI model override |
| `LLM_BACKEND` | ❌ | `openai` | `openai` or `ollama` |
| `OLLAMA_MODEL` | ❌ | `gemma4:e2b` | Ollama model name |
| `OLLAMA_BASE_URL` | ❌ | `http://localhost:11434` | Ollama server URL |
| `DUCKDB_PATH` | ❌ | `:memory:` | File path for persistent DuckDB |
| `PARQUET_STORE` | ❌ | `./data/parquet` | Directory for Parquet files |
| `LANGSMITH_API_KEY` | ❌ | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | ❌ | — | Set `true` to enable tracing |
