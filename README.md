# Efficient Text-to-SQL — DuckDB Analytics Agent

A **state-of-the-art multi-agent NL-to-SQL system** built with LangGraph, DuckDB, and OpenAI.
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

---

## Quick Start

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Python ≥ 3.11
- OpenAI API key
- [Ollama](https://ollama.com/) running locally (if using local models)

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

Open the URL printed in the terminal (usually `http://localhost:2024`) to chat with the agent in LangGraph Studio.

### 4. Load datasets and start chatting

```python
from agent.tools import load_file

load_file.invoke({"path": "data/sample_sales_1000.xlsx",  "dataset_name": "sales1000"})
load_file.invoke({"path": "data/sales_rep_targets.xlsx",  "dataset_name": "sales_rep_targets"})
load_file.invoke({"path": "data/product_metrics.xlsx",    "dataset_name": "product_metrics"})
```

Then ask in Studio:
> *"Which sales reps are below quota? Show actual revenue vs target and the gap."*
> *"Break down revenue by product category and compare to the baseline in product_metrics."*

---

## Project Structure

```
efficient-text-to-sql/
├── agent/
│   ├── __init__.py
│   ├── state.py            # AnalyticsState (Pydantic) + PlanStep
│   ├── database.py         # DuckDB singleton, metadata tables, schema indexing
│   ├── tools.py            # Deterministic @tool functions (lazy db dispatch)
│   ├── prompts.py          # System prompts per node
│   ├── nodes.py            # LangGraph node functions
│   └── graph.py            # StateGraph → exports `graph` for Studio
├── data/
│   ├── sample_sales_1000.xlsx
│   ├── sales_rep_targets.xlsx
│   └── product_metrics.xlsx
├── scripts/
│   ├── seed_multi_table.py     # Seed all three tables + register join keys
│   └── generate_test_data.py   # Regenerate synthetic test data
├── tests/
│   ├── test_multi_table.py             # Relationship detection, JOIN queries
│   ├── test_multi_table_seed.py        # Seed data integrity
│   ├── test_parquet_persistence.py     # Parquet + catalog correctness
│   ├── test_table_selection.py         # Keyword-based table routing
│   └── test_langgraph_query_rounds.py  # 40 live LLM round-trip queries
├── langgraph.json
├── pyproject.toml
├── .env.example
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

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Zero LLM code execution** | Safety + auditability — LLM outputs SQL strings only |
| **DuckDB internal metadata** | No external Postgres; all catalog tables live inside DuckDB |
| **Parquet persistence** | Each ingested file is stored as Parquet; views are re-attached on restart |
| **Lazy `sys.modules` dispatch** | Prevents stale binding when `agent.database` is reset (tests + restarts) |
| **MemorySaver checkpointer** | Multi-turn conversation in Studio without Redis/Postgres |
| **uv** | Fastest Python dependency resolver; single `uv sync` installs everything |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4o` | Model name override |
| `DUCKDB_PATH` | ❌ | `.local/duckdb/efficient-text-to-sql.duckdb` | Path to persistent DuckDB file |
| `PARQUET_STORE` | ❌ | `.local/parquet/` | Directory for Parquet files |
| `LANGSMITH_API_KEY` | ❌ | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | ❌ | — | Set to `true` to enable tracing |

---

## Upcoming — NIQ Data Integration

The next development phase targets real-world NIQ (NielsenIQ) retail datasets. Planned work:

- **NIQ schema adapters** — column normalisation and type mapping for NIQ export formats (retailer, period, measure, hierarchy columns)
- **Domain-aware semantic map** — pre-seed `_semantic_map` with NIQ business terms (`volume_sales`, `value_sales`, `weighted_distribution`, `numeric_distribution`, `SOM`, etc.)
- **Period handling** — NIQ uses non-standard period keys (e.g. `P01 2024`); dedicated date parser and DuckDB macro for period-to-date conversion
- **Hierarchy support** — product hierarchy (total market → category → subcategory → brand → SKU) and retail hierarchy (total → channel → banner → store) as first-class join dimensions
- **Measure catalogue** — explicit registry of derived measures (share of market, rate of sale, price index) with formulas stored in `_semantic_map` so the SQL Writer can reconstruct them without guessing
- **Multi-period comparisons** — rolling MAT, YTD, and vs-year-ago logic baked into the orchestrator planning step
