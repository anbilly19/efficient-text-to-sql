# Efficient Text-to-SQL — DuckDB Analytics Agent

A **state-of-the-art multi-agent NL-to-SQL system** built with LangGraph, DuckDB, and OpenAI.
Zero LLM code execution — the LLM only produces SQL strings; all computation is deterministic.

---

## How It Works

```
╔══════════════════════════════════════════════════════════════════╗
║                        USER QUESTION                            ║
║          "What was the revenue growth from 2023 to 2024?"       ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               │
                               ▼
             ┌─────────────────────────────────┐
             │          ORCHESTRATOR           │
             │                                 │
             │  • Understands the question     │
             │  • Breaks it into steps         │
             │  • Knows which tables exist     │
             │  • Writes the final answer      │
             └────────────┬────────────────────┘
                          │  Step-by-step plan
                          ▼
          ┌───────────────────────────────┐
          │           SQL WRITER          │
          │                              │
          │  • Sees exact column names   │
          │  • Generates a SELECT query  │
          │  • Handles date quirks auto  │
          └──────────────┬───────────────┘
                         │  SQL query
                         ▼
          ┌───────────────────────────────┐
          │         EXECUTE SQL           │
          │                              │
          │  • Runs query in DuckDB      │
          │  • No LLM involved here      │
          │  • Returns raw results       │
          └──────────────┬───────────────┘
                         │  Results
                         ▼
          ┌───────────────────────────────┐
          │           VERIFIER            │
          │                              │
          │  • Checks if answer is right │
          │  • Fixes the SQL if not      │◀──── retries automatically
          │  • Passes when confident     │
          └──────────────┬───────────────┘
                         │  Verified result
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║                        FINAL ANSWER                             ║
║       "Revenue grew by 18.4% from $1.2M in 2023 to $1.42M      ║
║        in 2024."                                                ║
╚══════════════════════════════════════════════════════════════════╝
```

### What each piece does

| Component | Role | Uses AI? |
|---|---|---|
| **Orchestrator** | Reads the question, builds a plan, writes the final human-readable answer | ✅ Yes |
| **SQL Writer** | Turns one step of the plan into a valid SQL query | ✅ Yes |
| **Execute SQL** | Runs the SQL against DuckDB — pure computation, no guessing | ❌ No |
| **Verifier** | Sanity-checks the result; rewrites the SQL if something looks wrong | ✅ Yes |

### Key ideas

- **The LLM never touches your data directly** — it only writes SQL strings. All computation is done by DuckDB.
- **Self-correcting** — if the SQL fails or looks wrong, the Verifier rewrites it and retries automatically.
- **Schema-grounded** — the SQL Writer sees your exact column names and sample values before writing anything, so it never guesses.
- **Date-aware** — date columns stored as text (common in Excel files) are detected and handled automatically.
- **Load any file** — Excel, CSV, or Parquet files load with a single natural language command.

---

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

### 4. Load a dataset and start asking questions

In the Studio chat:

```
load sample_sales_1000.xlsx as sales1000
```

Then ask anything:

```
What are the top 5 products by total revenue in Q4?
What was the month-by-month revenue trend for 2023?
Which customers placed orders in both 2022 and 2023?
```

---

## Project Structure

```
efficient-text-to-sql/
├── agent/
│   ├── __init__.py
│   ├── state.py        # AnalyticsState (Pydantic) + PlanStep
│   ├── database.py     # DuckDB singleton + _schema_catalog
│   ├── tools.py        # Deterministic @tool functions
│   ├── prompts.py      # System prompts for each node
│   ├── nodes.py        # LangGraph node functions
│   └── graph.py        # StateGraph → exports `graph` for Studio
├── tests/
│   └── test_langgraph_query_rounds.py   # 11-round regression suite
├── langgraph.json      # LangGraph Studio config
├── pyproject.toml      # uv project + dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Zero LLM code execution** | Safety + auditability — LLM outputs SQL strings only |
| **DuckDB internal metadata** | No external Postgres; `_schema_catalog` lives inside DuckDB |
| **Self-correcting verifier loop** | Bad SQL is automatically detected and rewritten |
| **Plain JSON responses** | LLM nodes return raw JSON text, not tool-call objects — more reliable across model versions |
| **MemorySaver checkpointer** | Multi-turn conversation in Studio without Redis/Postgres |
| **uv** | Fastest Python dependency resolver; single `uv sync` installs everything |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | Model name override |
| `DUCKDB_PATH` | ❌ | `:memory:` | Path to a persistent DuckDB file |
| `LANGSMITH_API_KEY` | ❌ | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | ❌ | — | Set to `true` to enable tracing |

---

## Running Tests

```bash
# Start LangGraph server first
uv run langgraph dev

# Then in a separate terminal
pytest -v tests/test_langgraph_query_rounds.py
```

The test suite runs all 11 query rounds (33 queries total) end-to-end against the live server.
