# Efficient Text-to-SQL — DuckDB Analytics Agent

A **multi-agent NL-to-SQL system** built with LangGraph, DuckDB, and OpenAI (or Ollama).
Zero LLM code execution — the LLM only produces SQL strings; all computation is deterministic.

---

## Current State

The system is **stable on `feat/niq-persistence`** with the following capabilities fully implemented:

| Capability | Status |
|---|---|
| Excel / CSV / Parquet ingestion → Parquet snapshot | ✅ Done |
| File-backed DuckDB with cold-start auto-reattach | ✅ Done |
| `_data_registry` — dataset inventory with row/column counts | ✅ Done |
| `_semantic_map` — business-term alias resolution | ✅ Done |
| Alias auto-seeding from `SCHEMA_CATALOG` on first start | ✅ Done |
| Pluggable LLM backend (OpenAI / Ollama) via CLI flag | ✅ Done |
| LangGraph Studio integration | ✅ Done |
| **Multi-table support (fact + summary + dimension tables)** | 🔲 Planned — see below |
| **Explicit join-key registry with verifier enforcement** | 🔲 Planned |
| **Table selection scoping (prompt size proportional to question)** | 🔲 Planned |

The primary dataset driving current development is `NIQ-Haushaltspaneldaten` — a German consumer household panel. The sales demo dataset (`sample_sales_1000.xlsx`) is used for multi-table development and testing.

---

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
- **SQL Writer** — generates precise DuckDB SQL (read-only `SELECT` only).
- **Execute SQL** — deterministic node, no LLM, runs the query.
- **Verifier** — cross-checks results with test queries and checksums.

---

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

---

## Persistence

By default the agent uses an **in-memory DuckDB** — all data is lost when the server stops.
To persist across restarts set these two env vars (or add to `.env`):

```env
DUCKDB_PATH=./data/niq.duckdb   # file-backed database
PARQUET_STORE=./data/parquet    # where ingested files are stored as Parquet
```

On next startup the agent:
1. Opens the same DuckDB file (metadata tables intact)
2. Re-attaches all Parquet views via `_data_registry`
3. Makes every previously loaded dataset immediately queryable — no re-upload needed

> **Note:** Setting `DUCKDB_PATH=:memory:` retains the in-memory default. Seeding and
> Parquet registration still run, but nothing persists past the process lifetime.

---

## Business Term Aliases

When a file is loaded the agent checks whether any column has a business-term alias
registered in `_semantic_map`. If none exist, a CSV template is written to
`data/templates/<dataset>_aliases.csv`.

Fill in the `column_name` column and register directly from the Studio chat:

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
The map is seeded once on first start from `SCHEMA_CATALOG` in `agent/schema_lookup.py`
and is never overwritten — human or agent edits survive restarts.

---

## Planned Development — Multi-Table Support

The next development phase adds principled multi-table support. The design treats this
as a **data model problem, not a prompt problem** — join knowledge and schema semantics
are stored as queryable DuckDB rows, not injected as narrative text.

### New metadata tables

Three new tables will be added to the DuckDB state store alongside the existing
`_data_registry` and `_semantic_map`:

**`_column_catalog`** — dataset-aware column registry with machine-readable role flags.
Replaces `schema_lookup.py` as a runtime dependency.

```sql
CREATE TABLE _column_catalog (
    dataset_name  VARCHAR NOT NULL,
    column_name   VARCHAR NOT NULL,
    display_name  VARCHAR,          -- English label
    description   VARCHAR,
    dtype         VARCHAR,
    is_metric     BOOLEAN,          -- aggregatable measure
    is_dimension  BOOLEAN,          -- filterable / groupable attribute
    is_join_key   BOOLEAN,          -- eligible as join key
    PRIMARY KEY (dataset_name, column_name)
);
```

**`_relationships`** — explicit join-key registry. Every JOIN the SQL Writer emits
is verified against this table by the Verifier. Unregistered joins are rejected.

```sql
CREATE TABLE _relationships (
    relationship_id  INTEGER PRIMARY KEY,
    left_table       VARCHAR NOT NULL,
    left_column      VARCHAR NOT NULL,
    right_table      VARCHAR NOT NULL,
    right_column     VARCHAR NOT NULL,
    join_type_hint   VARCHAR DEFAULT 'LEFT',  -- LEFT | INNER | MANY-TO-MANY
    cardinality      VARCHAR,                 -- e.g. 'N:1', '1:N', 'N:N'
    description      VARCHAR,
    UNIQUE (left_table, left_column, right_table, right_column)
);
```

**`_table_context`** — table-level grain, role, and prose description. Used by
the Orchestrator for lightweight table selection before injecting any schema —
keeping prompt size proportional to the active table set, not the total count.

```sql
CREATE TABLE _table_context (
    dataset_name  VARCHAR PRIMARY KEY,
    grain         VARCHAR,       -- e.g. 'one row per order'
    table_role    VARCHAR,       -- 'fact' | 'dimension' | 'summary' | 'lookup'
    description   VARCHAR,
    is_derived    BOOLEAN DEFAULT FALSE,
    derived_from  VARCHAR
);
```

### New tools (planned)

| Tool | Description |
|---|---|
| `list_tables_with_context` | All registered tables with grain, role, description |
| `get_schema_for_tables(table_names)` | Scoped schema + join relationships for selected tables only |
| `register_relationship(...)` | Insert a row into `_relationships` |
| `select_tables(question)` | LLM-callable table selector — returns relevant table subset |

`list_loaded_tables` will be retained for backwards compatibility but marked deprecated
in favour of `list_tables_with_context`.

### Derived table lifecycle (example: sales dataset)

```
load_file("sample_sales_1000.xlsx")
  → sales1000 registered as fact table (grain: one row per order)

create_derived_tables()
  → sales_rep_targets  (role: summary, grain: one row per rep-region)
      columns: sales_rep, region, orders_count, historical_revenue,
               units_sold, avg_order_value, annual_quota
  → product_metrics    (role: summary, grain: one row per product-category)
      columns: product, category, orders_count, total_units,
               total_revenue, avg_unit_price

  → _relationships seeded:
      sales1000.(sales_rep, region) → sales_rep_targets.(sales_rep, region)  [N:1]
      sales1000.product             → product_metrics.product                [N:1]
```

Derived tables and relationships are created idempotently — safe to call on every startup.

### Multi-table SQL Writer rules (planned)

When joining a fact table to a summary table the SQL Writer will be instructed to:

1. Only use tables returned by `select_tables` for the current question.
2. Only JOIN on key combinations registered in `_relationships`.
3. Aggregate the fact side to match the summary table's grain before joining — never SELECT a raw fact metric alongside a summary metric without aggregating first (fan-out guard).
4. Annotate each JOIN with a SQL comment referencing the registered relationship used.

### Verifier additions (planned)

- **Join registry check** — parse `ON` conditions from generated SQL and verify each pair exists in `_relationships`. Reject queries with unregistered joins.
- **Cardinality explosion check** — compare result row count against the fact table's `row_count` in `_data_registry`. Flag results that exceed the fact row count (indicates a fan-out join).

### Planned file changes

| # | File | Change |
|---|------|--------|
| 1 | `agent/database.py` | Add `_column_catalog`, `_relationships`, `_table_context`; `_infer_column_catalog`; `get_schema_context`; `create_derived_tables`; `select_relevant_tables` |
| 2 | `agent/state.py` | Add `active_tables: list[str]` and `schema_context: str` to `AnalyticsState` |
| 3 | `agent/tools.py` | Add `list_tables_with_context`, `get_schema_for_tables`, `register_relationship`, `select_tables`; deprecate `list_loaded_tables` |
| 4 | `agent/nodes.py` | Orchestrator: table selection → schema scoping → `state.active_tables` + `state.schema_context`; Verifier: join registry check + cardinality check |
| 5 | `agent/prompts.py` | Multi-table rules in `ORCHESTRATOR_SYSTEM` and `SQL_WRITER_SYSTEM` |
| 6 | `agent/schema_lookup.py` | Add `is_metric`, `is_dimension`, `is_join_key` booleans to every entry; retained as seed source only |
| 7 | `config/` | Add `sales_rep_targets_seed.csv`, `product_metrics_seed.csv`, `relationships_seed.csv` |
| 8 | `tests/test_multi_table.py` | Golden-query tests: join correctness, cardinality check, fan-out detection, table selection |

**Implementation order:** database.py → state.py → tools.py → nodes.py → prompts.py → schema_lookup.py → config seeds → tests

---

## Known Limitations

### Single-table semantics (current)

- **One active dataset assumed.** The Orchestrator injects schema context from the most recently loaded table. If multiple tables are loaded, only the last one is used for query planning.
- **No join awareness.** The SQL Writer has no information about relationships between tables. Cross-table queries require the user to describe the join explicitly in the question.
- **Schema context is fully injected.** The entire column alias list is added to every prompt regardless of the question's scope. For wide tables (>30 columns) this wastes context budget and can degrade generation quality.

### Persistence layer

- **Excel files are write-once.** After Parquet conversion the original Excel file is not re-read. Schema changes in the source file require re-ingesting and re-registering the dataset.
- **`schema_lookup.py` is statically coupled to NIQ.** The `SCHEMA_CATALOG` dict is NIQ-specific. Other datasets get no alias seeding unless entries are added manually or by agent. The planned `_column_catalog` migration will remove this coupling.
- **No schema drift detection.** If a Parquet file is regenerated with different columns, `_column_catalog` and `_semantic_map` are not automatically updated. A `validate_catalog_against_table()` utility is planned but not yet implemented.

### Multi-table (planned — not yet available)

- **Many-to-many joins** — the `_relationships` table will store a cardinality hint but the verifier will only flag fan-outs by row count, not by semantic correctness. True M:N relationships (e.g., orders ↔ promotions) require a bridge table registered explicitly.
- **>10 tables** — the planned `select_relevant_tables` function uses keyword pre-filtering over `_table_context.description`. Beyond ~10 tables a vector embedding lookup will be more reliable. The `select_tables` tool interface is designed to support this upgrade without changing callers.
- **Cross-database joins** — all tables are assumed to live in the same DuckDB file. Cross-file joins via DuckDB `ATTACH` are out of scope.

### LLM backend

- **Ollama quality gap.** Smaller local models (e.g., `gemma4:e2b`) produce correct SQL for simple single-table queries but are less reliable on multi-step plans, column alias resolution, and join queries. `gpt-4o-mini` or `gpt-4o` is recommended for production use.
- **No streaming output.** The agent returns a complete answer after all nodes have run. Intermediate plan steps are visible in LangGraph Studio but not in the CLI.

---

## LLM Backends

| Backend | Command | Notes |
|---|---|---|
| OpenAI (default) | `uv run cli.py` | Requires `OPENAI_API_KEY` |
| OpenAI custom model | `uv run cli.py --model gpt-4o` | |
| Ollama local | `uv run cli.py --backend ollama` | Default model: `gemma4:e2b` |
| Ollama custom model | `uv run cli.py --backend ollama --model gemma4:e4b` | |
| Ollama custom URL | `uv run cli.py --backend ollama --ollama-url http://host:11434` | |

Install Ollama: https://ollama.com/download — then `ollama pull gemma4:e2b`.

---

## Project Structure

```
efficient-text-to-sql/
├── agent/
│   ├── __init__.py
│   ├── state.py            # AnalyticsState (Pydantic) + PlanStep
│   ├── database.py         # DuckDB singleton, metadata tables, Parquet restore
│   ├── tools.py            # Deterministic @tool functions
│   ├── prompts.py          # System prompts for each node
│   ├── nodes.py            # LangGraph node functions + LLM factory
│   ├── schema_lookup.py    # NIQ column catalog — seed source for _semantic_map
│   └── graph.py            # StateGraph → exports `graph` for Studio
├── config/                 # Seed CSVs for metadata tables (planned)
├── cli.py                  # CLI wrapper — LLM backend as command-line arg
├── langgraph.json          # LangGraph Studio config
├── pyproject.toml          # uv project + dependencies
├── .env.example            # Environment variable template
└── README.md
```

---

## Tools Reference

| Tool | Description |
|---|---|
| `load_file` | Ingest Excel / CSV / Parquet → Parquet snapshot + DuckDB view |
| `list_loaded_tables` | Show all registered datasets (name, rows, source file) |
| `get_schema` | Column names, types, sample values for a dataset |
| `profile_column` | Null %, top-10 values, min/max/mean for a column |
| `run_sql` | Execute a read-only SELECT (first 50 rows returned) |
| `run_test_query` | Same as `run_sql` with checksums for Verifier |
| `lookup_semantic` | Resolve a business term → exact column name |
| `search_semantic_lookup` | Keyword search across column names and descriptions |
| `register_alias` | Add / update a business-term alias in `_semantic_map` |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Zero LLM code execution** | Safety + auditability — LLM outputs SQL strings only |
| **Parquet-backed views** | Minimal memory; datasets survive restarts without re-upload |
| **`_data_registry`** | Tracks every ingested file so views can be re-attached on cold start |
| **`_semantic_map`** | Business-term aliases decouple German column names from English queries |
| **Schema intelligence as data** | Planned column catalog and relationship registry keep schema knowledge queryable and versioned, not embedded in prompts |
| **Pluggable LLM backend** | Switch OpenAI ↔ Ollama with one CLI flag, no code changes |
| **uv** | Fastest Python dependency resolver; single `uv sync` installs everything |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ (OpenAI) | — | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | OpenAI model override |
| `LLM_BACKEND` | ❌ | `openai` | `openai` or `ollama` |
| `OLLAMA_MODEL` | ❌ | `gemma4:e2b` | Ollama model name |
| `OLLAMA_BASE_URL` | ❌ | `http://localhost:11434` | Ollama server URL |
| `DUCKDB_PATH` | ❌ | `:memory:` | File path for persistent DuckDB |
| `PARQUET_STORE` | ❌ | `./data/parquet` | Directory for Parquet snapshots |
| `LANGSMITH_API_KEY` | ❌ | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | ❌ | — | Set `true` to enable tracing |
