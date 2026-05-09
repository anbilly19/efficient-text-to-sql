# Efficient Text-to-SQL — DuckDB Analytics Agent

A **multi-agent NL-to-SQL system** built with LangGraph, DuckDB, and OpenAI (or Ollama).  
Zero LLM code execution — the LLM only produces SQL strings; all computation is deterministic.

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

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- Python ≥ 3.11
- OpenAI API key **or** [Ollama](https://ollama.com) running locally

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

Open the URL printed in the terminal (usually `http://localhost:2024`) to chat in LangGraph Studio.

### 4 — Load data and query

In the Studio chat:

```
load file at data/sales.xlsx as sales
```

Then ask:

> *What are the top 5 products by total revenue in Q4?

---

## Persistence

By default the agent uses an **in-memory DuckDB** — all data is lost when the server stops.  
For durable sessions set both env vars:

```env
DUCKDB_PATH=./data/niq.duckdb
PARQUET_STORE=./data/parquet
```

On restart the agent:
1. Opens the same DuckDB file (all metadata tables intact).
2. Re-attaches every Parquet view registered in `_data_registry`.
3. Makes every previously loaded dataset immediately queryable — no re-upload needed.

> **Note:** `DUCKDB_PATH=:memory:` is still supported for ephemeral sessions;  
> semantic edits will not persist past process lifetime.

---

## Business-Term Aliases

When a file is loaded the agent:
1. Populates `_semantic_lookup` with per-column statistics (type, cardinality, samples).
2. Writes a CSV template to `data/templates/<dataset>_aliases.csv` if no aliases exist yet.

Fill in the `definition_sql` column and register aliases:

```
# From Studio chat:
register alias umsatz = "Umsatz_EUR" — net revenue in EUR

# Or bulk-import the edited CSV:
from agent.tools import register_alias
```

Aliases stored in `_semantic_map` are used automatically by the SQL Writer.

---

## LLM Backends

| Backend | Command | Notes |
|---|---|---|
| OpenAI (default) | `uv run cli.py` | Requires `OPENAI_API_KEY` |
| OpenAI custom model | `uv run cli.py --model gpt-4o` | |
| Ollama local | `uv run cli.py --backend ollama` | Default: `gemma4:e2b` |
| Ollama custom model | `uv run cli.py --backend ollama --model gemma4:e4b` | |
| Ollama custom URL | `uv run cli.py --backend ollama --ollama-url http://host:11434` | |

Install Ollama: <https://ollama.com/download> — then `ollama pull gemma4:e2b`.

> **Model note:** `gemma4:e2b` is a ~2B-param model; tool-call reliability is lower than
> larger models. For production use prefer `gemma4:e4b` (4B) or a 27B variant.

---

## Project Structure

```
efficient-text-to-sql/
├── agent/
│   ├── __init__.py
│   ├── state.py          # AnalyticsState (Pydantic) + PlanStep
│   ├── database.py       # DuckDB singleton, metadata tables, Parquet restore
│   ├── tools.py          # Deterministic @tool functions
│   ├── prompts.py        # System prompts per node
│   ├── nodes.py          # LangGraph node functions + LLM factory
│   └── graph.py          # StateGraph → exports `graph` for Studio
├── cli.py                # CLI wrapper — LLM backend as command-line arg
├── langgraph.json        # LangGraph Studio config
├── pyproject.toml        # uv project + dependencies
├── .env.example          # Environment variable template
└── README.md
```

---

## Tools Reference

| Tool | Description |
|---|---|
| `load_file` | Ingest Excel / CSV / Parquet → Parquet + DuckDB view |
| `list_loaded_tables` | Show all registered datasets (name, rows, source file) |
| `get_schema` | Column names, types, sample values for a dataset |
| `profile_column` | Null %, top-10 values, min/max/mean for a column |
| `run_sql` | Execute a read-only SELECT (first 50 rows) |
| `run_test_query` | Same as `run_sql` with checksums for verification |
| `lookup_semantic` | Resolve a business term to a SQL expression |
| `search_semantic_lookup` | Keyword search across column names and descriptions |
| `register_alias` | Add / update a business-term alias in `_semantic_map` |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ (OpenAI) | — | OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | OpenAI model override |
| `LLM_BACKEND` | ❌ | `openai` | `openai` or `ollama` |
| `OLLAMA_MODEL` | ❌ | `gemma4:e2b` | Ollama model name |
| `OLLAMA_BASE_URL` | ❌ | `http://localhost:11434` | Ollama server URL |
| `DUCKDB_PATH` | ❌ | `:memory:` | File path for persistent DuckDB |
| `PARQUET_STORE` | ❌ | `./data/parquet` | Directory for Parquet files |
| `LANGSMITH_API_KEY` | ❌ | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | ❌ | — | Set `true` to enable tracing |

---

## Planned Development

### Dynamic schema discovery at ingest time

Currently the agent has no automatic understanding of what a column *means* — it relies entirely
on human-curated alias maps or the LLM guessing from raw column names at query time. The next
major milestone replaces this with a structured discovery pass run once per dataset:

1. **Role classification** — pure heuristics on name, type, and cardinality tag every column as
   one of `measure`, `dimension`, `time_key`, `identifier`, or `free_text`. No LLM required.
2. **LLM-assisted label proposal** — a single structured LLM call at ingest time (not query time)
   receives column names + sample values and returns proposed English labels and synonyms.
   Output is written to `_semantic_map` with `status = 'proposed'`.
3. **Human review loop** — the agent reports how many columns were auto-labelled and how many
   need review, writes a pre-filled CSV template, and awaits explicit confirmation:
   ```
   Profiled 22 columns. 18 labels proposed automatically.
   Review: data/templates/my_dataset_aliases.csv
   Confirm with: confirm aliases my_dataset
   ```
4. **`confirmed` flag in `_semantic_map`** — the SQL Writer preferentially uses confirmed aliases;
   when only proposed aliases are available it hedges in the response.
   A `list_pending_aliases` tool surfaces all unconfirmed mappings.

### Human input formats accepted at ingest

Beyond the auto-generated CSV template, the agent will accept:

| Input | Description |
|---|---|
| **CSV alias template** (auto-generated) | Pre-filled from discovery; analyst edits and re-drops |
| **External data dictionary** | A second file with `column_name`, `description`, `business_term` columns — common in corporate BI |
| **JSON schema file** | `{"column": {"label": "...", "aliases": ["..."]}}` — easy for dev-authored datasets |
| **Plain-text glossary** | A `.txt` / `.md` doc; agent extracts alias mappings via a one-shot LLM parse |
| **Exported `_semantic_map` CSV** | Re-seed a fresh deployment from a previous session's confirmed aliases |

### Other planned improvements

- **`graph.py` bug fix** — `load_file_node` is now wired; remaining edge-case: the
  `route_after_orchestrator` guard does not yet handle `load_file_path` being stale
  across turns when a previous load failed mid-way.
- **Multi-table queries** — orchestrator currently plans against a single active table;
  join-aware planning across multiple registered datasets is not yet supported.
- **Incremental file updates** — appending new rows to an existing Parquet dataset without
  full re-ingest.
- **Alias learning from successful queries** — log column usages during verified queries and
  propose new aliases when a pattern repeats; requires human confirmation before promotion.
- **Export** — `run_sql` returns up to 50 rows; larger result exports (CSV download) not yet
  available from the chat interface.

---

## Known Limitations

| Area | Limitation |
|---|---|
| **Alias coverage** | Without a curated `_semantic_map`, the agent must infer column intent from raw names and samples. On datasets with opaque or foreign-language column names (e.g. German panel data), SQL quality degrades noticeably until aliases are confirmed. |
| **Single-table planning** | The orchestrator builds plans against one dataset at a time. Cross-table joins require manual SQL or are silently ignored. |
| **`load_file_node` routing** | The `route_after_orchestrator` function checks `state.load_file_path` but does not clear it on re-entry after a failed load; a second `load` command in the same thread may be silently ignored. |
| **Ollama tool-call reliability** | Small models (≤ 4B) frequently mis-format tool calls or skip required fields. The agent degrades to a best-effort SQL attempt rather than failing gracefully. |
| **Memory mode has no persistence** | `DUCKDB_PATH=:memory:` means all loaded tables, schema catalog, and semantic aliases are lost when the process exits. |
| **Row limit on results** | `run_sql` returns a maximum of 50 rows. Aggregation queries are unaffected, but queries that need full result sets (e.g. exports) are not supported via the chat interface. |
| **No incremental ingest** | Re-loading a file under the same `dataset_name` drops and replaces the table; there is no append mode. |
| **Static NIQ catalog** | The original `schema_lookup.py` / `SCHEMA_CATALOG` is NIQ-specific and still imported in some code paths. It should be treated as an optional seed, not a core dependency; refactoring is tracked as a cleanup task. |
| **Verifier checksums** | The verifier uses row-count and checksum heuristics; it does not perform semantic validation (e.g. detecting plausible-but-wrong aggregation logic). |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Zero LLM code execution | Safety + auditability — LLM outputs SQL strings only |
| Parquet-backed views | Minimal memory footprint; datasets survive restarts without re-upload |
| `_data_registry` | Tracks every ingested file so views can be re-attached on cold start |
| `_semantic_map` | Business-term aliases decouple foreign/opaque column names from English queries |
| `proposed` / `confirmed` alias states *(planned)* | LLM suggestions are never auto-trusted; human confirmation gates production use |
| Pluggable LLM backend | Switch OpenAI ↔ Ollama with one CLI flag, no code changes |
| uv | Fastest Python dependency resolver; single `uv sync` installs everything |
