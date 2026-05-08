"""LangGraph node functions – one per agent role."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts import (
    PROFILER_SYSTEM,
    SQL_WRITER_SYSTEM,
    VERIFIER_SYSTEM,
)
from agent.state import AnalyticsState, PlanStep
from agent.tools import (
    ORCHESTRATOR_TOOLS,
    PROFILER_TOOLS,
    SQL_WRITER_TOOLS,
    get_schema,
    load_file,
    run_sql,
)
from agent.database import get_connection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(p for p in parts if p).strip()
    return str(content)


def _try_parse_json(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    best = None
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        best = parsed
                except json.JSONDecodeError:
                    pass
                start = None
    return best


def _get_available_tables() -> str:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT dataset_name, column_name, data_type FROM _schema_catalog ORDER BY dataset_name, column_name"
        ).fetchall()
        if not rows:
            return "No tables loaded yet."
        tables: dict[str, list[str]] = {}
        for dataset, col, dtype in rows:
            tables.setdefault(dataset, []).append(f"{col} ({dtype})")
        lines = []
        for tname, cols in tables.items():
            lines.append(f"Table: {tname}")
            lines.append("  Columns: " + ", ".join(cols))
        return "\n".join(lines)
    except Exception as exc:
        return f"(Could not read schema catalog: {exc})"


def _get_most_recent_table() -> str:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT dataset_name FROM _schema_catalog ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else ""
    except Exception:
        return ""


def _get_schema_for_table(table_name: str) -> str:
    try:
        raw = get_schema.invoke({"dataset": table_name})
        cols = json.loads(raw)
        lines = [f"Table: {table_name}"]
        for col in cols:
            samples = ", ".join(repr(s) for s in col.get("samples", []))
            lines.append(f"  - \"{col['column']}\" ({col['type']})  samples: [{samples}]")
        return "\n".join(lines)
    except Exception as exc:
        return f"(Schema unavailable: {exc})"


def _get_varchar_date_columns(table_name: str) -> set[str]:
    if not table_name:
        return set()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT column_name, data_type FROM _schema_catalog WHERE dataset_name = ?",
            [table_name],
        ).fetchall()
    except Exception:
        return set()
    result = set()
    for col, dtype in rows:
        if dtype.upper() == "VARCHAR":
            try:
                sample = conn.execute(
                    f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 1'
                ).fetchone()
                if sample and re.match(r"\d{4}-\d{2}-\d{2}", str(sample[0])):
                    result.add(col)
            except Exception:
                pass
    return result


def _get_date_cast_warnings(table_name: str) -> str:
    if not table_name:
        return ""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT column_name, data_type FROM _schema_catalog WHERE dataset_name = ?",
            [table_name],
        ).fetchall()
    except Exception:
        return ""
    warnings: list[str] = []
    for col, dtype in rows:
        if dtype.upper() == "VARCHAR":
            try:
                sample = conn.execute(
                    f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 1'
                ).fetchone()
                if sample and re.match(r"\d{4}-\d{2}-\d{2}", str(sample[0])):
                    warnings.append(
                        f'  ⚠️  "{col}" is VARCHAR storing dates (e.g. \'{sample[0]}\').'
                        f' NEVER use YEAR("{col}") or MONTH("{col}") directly.'
                        f' ALWAYS write: YEAR(TRY_CAST("{col}" AS DATE))'
                        f' / MONTH(TRY_CAST("{col}" AS DATE))'
                        f" / DATE_TRUNC('month', TRY_CAST(\"{col}\" AS DATE))"
                    )
            except Exception:
                pass
        elif dtype.upper() == "DATE":
            warnings.append(
                f'  ✅  "{col}" is DATE — use YEAR("{col}"), MONTH("{col}"), DATE_TRUNC directly.'
            )
    return "\n".join(warnings)


def _sanitize_sql(sql: str, table_name: str) -> str:
    if not sql:
        return sql
    # Fix stray double-quotes: "col"" -> "col"
    sql = re.sub(r'"(\w+)""', r'"\1"', sql)
    # Fix leading double-quote on identifier: ""col" -> "col"
    sql = re.sub(r'""(\w+)"', r'"\1"', sql)
    for col in _get_varchar_date_columns(table_name):
        col_pattern = rf'(?:"?\w+"?\.)?\"?{re.escape(col)}\"?'
        sql = re.sub(
            rf'\bYEAR\s*\(\s*({col_pattern})\s*\)',
            lambda m, c=col: f'YEAR(TRY_CAST("{c}" AS DATE))',
            sql, flags=re.IGNORECASE
        )
        sql = re.sub(
            rf'\bMONTH\s*\(\s*({col_pattern})\s*\)',
            lambda m, c=col: f'MONTH(TRY_CAST("{c}" AS DATE))',
            sql, flags=re.IGNORECASE
        )
        sql = re.sub(
            rf"DATE_TRUNC\s*\(\s*('[^']*')\s*,\s*({col_pattern})\s*\)",
            lambda m, c=col: f"DATE_TRUNC({m.group(1)}, TRY_CAST(\"{c}\" AS DATE))",
            sql, flags=re.IGNORECASE
        )
        sql = re.sub(
            rf"STRFTIME\s*\(\s*('[^']*')\s*,\s*({col_pattern})\s*\)",
            lambda m, c=col: f"STRFTIME({m.group(1)}, TRY_CAST(\"{c}\" AS DATE))",
            sql, flags=re.IGNORECASE
        )
    return sql


def _next_pending_step(plan: list[PlanStep]) -> PlanStep | None:
    return next((s for s in plan if s.status == "pending"), None)


_LOAD_PATTERNS = [
    re.compile(r"load\s+(?:file\s+)?at\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"import\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"load\s+(?P<path>\S+\.(?:xlsx|xls|csv|parquet))\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"load\s+(?:file\s+)?(?P<path>\S+\.(?:xlsx|xls|csv|parquet))", re.IGNORECASE),
]

# Patterns that indicate the user is asking about table schema / columns
_SCHEMA_PATTERNS = [
    re.compile(r"\bwhat\s+columns?\b", re.IGNORECASE),
    re.compile(r"\blist\s+columns?\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:available\s+)?(?:columns?|fields?|schema)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:fields?|attributes?|schema)\b", re.IGNORECASE),
    re.compile(r"\bdescribe\s+(?:the\s+)?table\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s|\s+is)\s+(?:in|inside)\s+(?:the\s+)?\w+\s+table\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?:columns?|fields?)\b", re.IGNORECASE),
    re.compile(r"\btable\s+structure\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+tables?\s+(?:do\s+(?:i|we|you)\s+have|are\s+(?:available|loaded))\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?(?:all\s+)?(?:available\s+)?tables?\b", re.IGNORECASE),
]


def _detect_load_intent(text: str) -> dict | None:
    for pattern in _LOAD_PATTERNS:
        m = pattern.search(text)
        if m:
            path = m.group("path")
            try:
                dataset = m.group("dataset")
            except IndexError:
                dataset = Path(path).stem.replace(" ", "_").replace("-", "_").lower()
            return {"path": path, "dataset": dataset}
    return None


def _detect_schema_intent(text: str) -> bool:
    """Return True if the user is asking about table schema/columns."""
    return any(p.search(text) for p in _SCHEMA_PATTERNS)


def _extract_table_from_plan(plan: list[PlanStep], user_query: str) -> str:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT DISTINCT dataset_name FROM _schema_catalog").fetchall()
        tables = [r[0] for r in rows]
        if not tables:
            return ""
        if len(tables) == 1:
            return tables[0]
        search_text = user_query.lower()
        for step in plan:
            search_text += " " + step.description.lower()
        matched = [t for t in tables if t.lower() in search_text]
        if matched:
            return max(matched, key=len)
        return _get_most_recent_table() or tables[0]
    except Exception:
        return ""


def _rows_to_markdown(rows_json: str) -> str:
    """Convert JSON rows string to markdown table. Returns raw string on any error."""
    if not rows_json or rows_json.startswith("ERROR"):
        return rows_json or "No result."
    try:
        rows = json.loads(rows_json)
        if not rows:
            return "No rows returned."
        headers = list(rows[0].keys())
        header_row = " | ".join(headers)
        sep_row = " | ".join(["---"] * len(headers))
        data_rows = "\n".join(
            " | ".join(str(row.get(h, "")) for h in headers) for row in rows
        )
        return f"{header_row}\n{sep_row}\n{data_rows}"
    except Exception:
        return rows_json


# ---------------------------------------------------------------------------
# Node: orchestrator
# ---------------------------------------------------------------------------

def orchestrator(state: AnalyticsState) -> dict:
    current_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            current_query = _extract_text(msg.content)
            break

    tables_context = _get_available_tables()
    most_recent_table = _get_most_recent_table()

    base_reset: dict = {
        "user_query": current_query,
        "final_answer": "",
        "plan": [],
        "current_step": None,
        "last_sql": "",
        "last_query_result": "",
        "last_query_metadata": {},
        "verification_verdict": "",
        "verification_feedback": "",
        "retry_count": 0,
        "load_file_path": None,
        "load_file_dataset": None,
        "error": "",
    }

    # ── Synthesis: fire when ALL steps are done or failed ──────────────────
    if state.plan and all(s.status in ("done", "failed") for s in state.plan):
        all_failed = all(s.status == "failed" for s in state.plan)

        if all_failed:
            err = state.error or "The query could not be executed."
            short_err = err.replace("ERROR: ", "").strip()
            final = (
                f"I wasn't able to answer that question due to a SQL error:\n\n"
                f"```\n{short_err}\n```\n\n"
                f"Could you rephrase, or check that the column names are correct?"
            )
        else:
            row_count = state.last_query_metadata.get("row_count", "?")
            table_preview = _rows_to_markdown(state.last_query_result)
            synthesis_prompt = (
                f"User question: {state.user_query}\n\n"
                f"SQL executed: {state.last_sql}\n\n"
                f"Row count: {row_count}\n\n"
                f"Query results:\n{table_preview}\n\n"
                "Write a clear, concise answer for the user. "
                "If the result is a table, present it as markdown. "
                "If it is a single value, describe it in plain English. "
                'Output ONLY this JSON: {"final_answer": "<your answer>"}'
            )
            response = _llm().invoke([HumanMessage(content=synthesis_prompt)])
            raw = _extract_text(response.content)
            parsed = _try_parse_json(raw) or {}
            final_val = parsed.get("final_answer")
            if final_val is None or not isinstance(final_val, str):
                final_val = raw if isinstance(raw, str) and raw.strip() else table_preview
            final = str(final_val).strip()

        return {
            **base_reset,
            "final_answer": final,
            "messages": [AIMessage(content=final)],
        }

    # ── Fast-path: regex load detection ───────────────────────────────────
    load_match = _detect_load_intent(current_query)
    if load_match:
        return {
            **base_reset,
            "load_file_path": load_match["path"],
            "load_file_dataset": load_match["dataset"],
        }

    # ── Fast-path: schema / column questions ──────────────────────────────
    if _detect_schema_intent(current_query):
        conn = get_connection()
        try:
            all_table_rows = conn.execute(
                "SELECT DISTINCT dataset_name FROM _schema_catalog"
            ).fetchall()
            all_tables = [r[0] for r in all_table_rows]
        except Exception:
            all_tables = []

        target_table: str | None = None
        query_lower = current_query.lower()
        for t in all_tables:
            if t.lower() in query_lower:
                target_table = t
                break
        if target_table is None and most_recent_table:
            target_table = most_recent_table

        if not all_tables:
            reply = "No tables are loaded yet. Load a file first with: `load file at <path> as <name>`."
        elif target_table:
            schema_text = _get_schema_for_table(target_table)
            reply = f"Here is the schema for the **{target_table}** table:\n\n```\n{schema_text}\n```"
        else:
            reply = f"Here are all loaded tables:\n\n```\n{tables_context}\n```"

        return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}

    # ── LLM planning ──────────────────────────────────────────────────────
    ORCHESTRATOR_SYSTEM = f"""You are the Orchestrator of a DuckDB analytics agent.

Available tables in DuckDB:
{tables_context}

Most recently loaded table: {most_recent_table or 'none'}

Classify the user message into exactly ONE intent:

1. ANALYTICS – use when the question asks about data IN a loaded table.
   This includes:
   - Counts, sums, averages, filters, rankings, trends
   - Questions about column values, unique values, or data distribution
   - Schema / column questions (e.g. "what columns does X have?", "show me the fields")
   {{"intent": "analytics", "plan": [{{"id": 1, "type": "sql", "description": "..."}}]}}
   Step types: "profile" (explore a column) or "sql" (SELECT query).
   Every step description MUST name the exact table.
   Default table if unspecified: "{most_recent_table}".

2. CHITCHAT – use for EVERYTHING else:
   - Greetings, thanks, how-are-you
   - General knowledge / world facts (not about the loaded data)
   - Questions about the agent itself
   {{"intent": "chitchat", "final_answer": "<helpful reply>"}}

Decision rule: if the question references a loaded table or asks about data/columns, pick ANALYTICS.
Output ONLY the JSON. No markdown, no explanation.
"""

    # Plain LLM — no tool binding so response is always plain text JSON
    response = _llm().invoke(
        [SystemMessage(content=ORCHESTRATOR_SYSTEM), HumanMessage(content=current_query)]
    )
    raw_text = _extract_text(response.content)
    parsed = _try_parse_json(raw_text)

    # Fallback: JSON parse failed but a table is loaded → build a default analytics plan
    if parsed is None and most_recent_table:
        parsed = {
            "intent": "analytics",
            "plan": [{"id": 1, "type": "sql",
                      "description": f"Answer the user question against the {most_recent_table} table: {current_query}"}],
        }

    if parsed is None:
        reply = raw_text or "Hi! Ask a data question or say: load file at <path> as <name>."
        return {**base_reset, "final_answer": str(reply), "messages": [AIMessage(content=str(reply))]}

    intent = parsed.get("intent", "chitchat")
    if intent == "analytics":
        raw_steps = parsed.get("plan", [])
        # Fallback: analytics intent but empty plan
        if not raw_steps and most_recent_table:
            raw_steps = [{"id": 1, "type": "sql",
                          "description": f"Answer the user question against the {most_recent_table} table: {current_query}"}]
        plan = [PlanStep(**step) for step in raw_steps]
        return {**base_reset, "plan": plan}

    reply = str(parsed.get("final_answer", "Hi! Ask a data question or load a file."))
    return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# Node: load_file_node
# ---------------------------------------------------------------------------

def load_file_node(state: AnalyticsState) -> dict:
    path = state.load_file_path
    dataset = state.load_file_dataset
    if not path:
        reply = "I couldn't find a file path. Please say: load file at <path> as <name>."
        return {"final_answer": reply, "messages": [AIMessage(content=reply)],
                "load_file_path": None, "load_file_dataset": None}
    if not dataset:
        dataset = Path(path).stem.replace(" ", "_").replace("-", "_").lower()
    result = load_file.invoke({"path": path, "dataset_name": dataset})
    reply = f"❌ Failed to load file: {result}" if result.startswith("ERROR") else (
        f"✅ {result}\n\nYou can now ask questions about the `{dataset}` table!"
    )
    return {"final_answer": reply, "messages": [AIMessage(content=reply)],
            "load_file_path": None, "load_file_dataset": None}


# ---------------------------------------------------------------------------
# Node: profiler
# ---------------------------------------------------------------------------

def profiler(state: AnalyticsState) -> dict:
    step = _next_pending_step(state.plan)
    if step is None:
        return {"current_step": None}
    llm = _llm().bind_tools(PROFILER_TOOLS)
    response = llm.invoke([
        SystemMessage(content=PROFILER_SYSTEM),
        HumanMessage(content=f"Profiling task: {step.description}\nUser question: {state.user_query}"),
    ])
    result_text = response.content or "No profiling result."
    updated_plan = [
        s.model_copy(update={"status": "done", "result": result_text})
        if s.id == step.id else s for s in state.plan
    ]
    return {"plan": updated_plan, "current_step": None}


# ---------------------------------------------------------------------------
# Node: sql_writer
# ---------------------------------------------------------------------------

def sql_writer(state: AnalyticsState) -> dict:
    step = _next_pending_step(state.plan)
    if step is None:
        step = next((s for s in state.plan if s.status == "running"), None)
    if step is None:
        return {"last_sql": "", "error": "No pending step found for sql_writer."}

    table_name = _extract_table_from_plan(state.plan, state.user_query)
    schema_context = _get_schema_for_table(table_name) if table_name else "(No tables loaded)"
    cast_warnings = _get_date_cast_warnings(table_name)

    # On retry, if verifier supplied a corrected SQL, use it directly
    if state.retry_count > 0 and state.last_sql and state.last_sql.strip().upper().startswith("SELECT"):
        sql = _sanitize_sql(state.last_sql, table_name)
        updated_plan = [
            s.model_copy(update={"status": "running"})
            if s.id == step.id else s for s in state.plan
        ]
        return {"last_sql": sql, "plan": updated_plan, "verification_feedback": "", "current_step": step}

    # Include results from ALL completed prior steps
    prior_step_notes = "\n".join(
        f"- Step {s.id} ({s.description}): {s.result}"
        for s in state.plan
        if s.status == "done" and s.result and s.id != step.id
    )

    prompt = (
        f"Sub-task: {step.description}\n"
        f"User question: {state.user_query}\n\n"
        f"=== EXACT SCHEMA (use these column names verbatim) ===\n"
        f"{schema_context}\n\n"
        f"=== MANDATORY DATE COLUMN RULES (follow exactly, no exceptions) ===\n"
        f"{cast_warnings if cast_warnings else '  (no date columns require special handling)'}\n\n"
        f"Prior step results (use if this step depends on earlier results):\n"
        f"{prior_step_notes or 'None'}\n\n"
        f"Verifier feedback (if retry): {state.verification_feedback or 'None'}\n\n"
        'Output JSON: {"sql": "...", "explanation": "..."}'
    )

    # Plain LLM — no tool binding so response is always plain text JSON
    response = _llm().invoke(
        [SystemMessage(content=SQL_WRITER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(_extract_text(response.content)) or {}
    sql = parsed.get("sql", "")

    # Guard: empty SQL — mark step failed immediately
    if not sql or not sql.strip():
        updated_plan = [
            s.model_copy(update={"status": "failed", "result": "sql_writer produced empty SQL"})
            if s.id == step.id else s for s in state.plan
        ]
        return {
            "last_sql": "",
            "plan": updated_plan,
            "error": "sql_writer produced empty SQL",
            "current_step": step,
        }

    sql = _sanitize_sql(sql, table_name)

    updated_plan = [
        s.model_copy(update={"status": "running"})
        if s.id == step.id else s for s in state.plan
    ]
    return {"last_sql": sql, "plan": updated_plan, "verification_feedback": "", "current_step": step}


# ---------------------------------------------------------------------------
# Node: execute_sql
# ---------------------------------------------------------------------------

def execute_sql(state: AnalyticsState) -> dict:
    # Guard: empty SQL
    if not state.last_sql or not state.last_sql.strip():
        def _fail_plan() -> list[PlanStep]:
            return [
                s.model_copy(update={"status": "failed", "result": "empty SQL"})
                if state.current_step and s.id == state.current_step.id else s
                for s in state.plan
            ]
        return {
            "last_query_result": "ERROR: empty SQL",
            "last_query_metadata": {},
            "plan": _fail_plan(),
            "error": "empty SQL",
        }

    result = run_sql.invoke({"query": state.last_sql})

    def _updated_plan(status: str, msg: str = "") -> list[PlanStep]:
        return [
            s.model_copy(update={"status": status, "result": msg})
            if state.current_step and s.id == state.current_step.id else s
            for s in state.plan
        ]

    if result.startswith("ERROR"):
        return {
            "last_query_result": result,
            "last_query_metadata": {},
            "plan": _updated_plan("failed", result),
            "error": result,
        }

    parsed = json.loads(result)
    return {
        "last_query_result": json.dumps(parsed.get("rows", []), default=str),
        "last_query_metadata": parsed.get("metadata", {}),
        "plan": _updated_plan("done"),
        "error": "",
    }


# ---------------------------------------------------------------------------
# Node: verifier — handles both success verification and SQL error recovery
# ---------------------------------------------------------------------------

def verifier(state: AnalyticsState) -> dict:
    step = state.current_step
    is_error = state.last_query_result.startswith("ERROR") if state.last_query_result else False
    row_count = state.last_query_metadata.get("row_count", "unknown")
    table_preview = _rows_to_markdown(state.last_query_result)

    if is_error:
        prompt = (
            f"Sub-task: {step.description if step else 'unknown'}\n"
            f"SQL that failed:\n{state.last_sql}\n\n"
            f"DuckDB error:\n{state.last_query_result}\n\n"
            f"Schema hint: {_get_schema_for_table(_get_most_recent_table())}\n\n"
            "The SQL produced an error. Provide a corrected SQL query.\n"
            'Output ONLY: {"verdict": "fail", "feedback": "<what was wrong>", "corrected_sql": "<fixed SQL>"}'
        )
    else:
        prompt = (
            f"Sub-task: {step.description if step else 'unknown'}\n"
            f"SQL executed:\n{state.last_sql}\n\n"
            f"Row count: {row_count}\n"
            f"First rows:\n{table_preview[:1500]}\n\n"
            "Verify whether the SQL correctly answers the sub-task.\n"
            "- Correct and returns data → verdict=pass\n"
            "- Clear bug (wrong column, bad filter) → verdict=fail with corrected_sql\n"
            "- Plausible but uncertain → verdict=warning (treated as pass)\n"
            "- Do NOT call any tools.\n"
            'Output ONLY: {"verdict": "pass|fail|warning", "feedback": "...", "corrected_sql": "(only if fail)"}'
        )

    response = _llm().invoke(
        [SystemMessage(content=VERIFIER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(_extract_text(response.content)) or {"verdict": "pass", "feedback": ""}
    verdict = str(parsed.get("verdict", "pass"))
    feedback = str(parsed.get("feedback", ""))
    corrected_sql = parsed.get("corrected_sql", "")

    new_status = "done" if verdict in ("pass", "warning") else "pending"
    updated_plan = [
        s.model_copy(update={"status": new_status, "result": feedback})
        if step and s.id == step.id else s for s in state.plan
    ]
    updates: dict = {
        "verification_verdict": verdict,
        "verification_feedback": feedback,
        "plan": updated_plan,
        "current_step": None,
        "retry_count": state.retry_count + 1,
    }
    if corrected_sql and verdict == "fail":
        updates["last_sql"] = str(corrected_sql)
    return updates
