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
                        f" NEVER use YEAR(\"{col}\") or MONTH(\"{col}\") directly."
                        f" ALWAYS write: YEAR(TRY_CAST(\"{col}\" AS DATE))"
                        f" / MONTH(TRY_CAST(\"{col}\" AS DATE))"
                        f" / DATE_TRUNC(\'month\', TRY_CAST(\"{col}\" AS DATE))"
                    )
            except Exception:
                pass
        elif dtype.upper() == "DATE":
            warnings.append(
                f'  ✅  "{col}" is DATE — use YEAR("{col}"), MONTH("{col}"), DATE_TRUNC directly.'
            )
    return "\n".join(warnings)


def _next_pending_step(plan: list[PlanStep]) -> PlanStep | None:
    return next((s for s in plan if s.status == "pending"), None)


_LOAD_PATTERNS = [
    re.compile(r"load\s+(?:file\s+)?at\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"import\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"load\s+(?P<path>\S+\.(?:xlsx|xls|csv|parquet))\s+as\s+(?P<dataset>\w+)", re.IGNORECASE),
    re.compile(r"load\s+(?:file\s+)?(?P<path>\S+\.(?:xlsx|xls|csv|parquet))", re.IGNORECASE),
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

    # Synthesis: all plan steps completed
    if state.plan and all(s.status in ("done", "failed") for s in state.plan):
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
        final = parsed.get("final_answer")
        if final is None or not isinstance(final, str):
            final = raw if isinstance(raw, str) and raw.strip() else table_preview
        return {
            **base_reset,
            "final_answer": str(final).strip(),
            "messages": [AIMessage(content=str(final).strip())],
        }

    # Fast-path: regex load detection
    load_match = _detect_load_intent(current_query)
    if load_match:
        return {
            **base_reset,
            "load_file_path": load_match["path"],
            "load_file_dataset": load_match["dataset"],
        }

    ORCHESTRATOR_SYSTEM = f"""You are the Orchestrator of a DuckDB analytics agent.

Available tables in DuckDB:
{tables_context}

Most recently loaded table: {most_recent_table or 'none'}

Classify the user message into exactly ONE intent:

1. ANALYTICS – use ONLY when the question explicitly asks about data IN a loaded table
   (counts, sums, averages, filters, rankings, trends over rows/columns listed above).
   {{"intent": "analytics", "plan": [{{"id": 1, "type": "sql", "description": "..."}}]}}
   Step types: "profile" (explore a column) or "sql" (SELECT query).
   Every step description MUST name the exact table.
   Default table if unspecified: "{most_recent_table}".

2. CHITCHAT – use for EVERYTHING else:
   - Greetings, thanks, how-are-you
   - General knowledge / world facts ("what is X?", "how does Y work?", comparisons
     between concepts that are NOT columns/tables)
   - Ambiguous questions with no clear table reference
   - Questions about the agent itself
   {{"intent": "chitchat", "final_answer": "<helpful reply>"}}

Decision rule: if in doubt, pick CHITCHAT.

Output ONLY the JSON. No markdown, no explanation.
"""

    llm = _llm().bind_tools(ORCHESTRATOR_TOOLS)
    response = llm.invoke(
        [SystemMessage(content=ORCHESTRATOR_SYSTEM), HumanMessage(content=current_query)]
    )
    parsed = _try_parse_json(response.content)
    if parsed is None:
        reply = response.content or "Hi! Ask a data question or say: load file at <path> as <name>."
        return {**base_reset, "final_answer": str(reply), "messages": [AIMessage(content=str(reply))]}

    intent = parsed.get("intent", "chitchat")
    if intent == "analytics":
        plan = [PlanStep(**step) for step in parsed.get("plan", [])]
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

    profiling_notes = "\n".join(
        f"- {s.description}: {s.result}"
        for s in state.plan
        if s.type == "profile" and s.status == "done" and s.result
    )

    prompt = (
        f"Sub-task: {step.description}\n"
        f"User question: {state.user_query}\n\n"
        f"=== EXACT SCHEMA (use these column names verbatim) ===\n"
        f"{schema_context}\n\n"
        f"=== MANDATORY DATE COLUMN RULES (follow exactly, no exceptions) ===\n"
        f"{cast_warnings if cast_warnings else '  (no date columns require special handling)'}\n\n"
        f"Profiling notes:\n{profiling_notes or 'None'}\n\n"
        f"Verifier feedback (if retry): {state.verification_feedback or 'None'}\n\n"
        'Output JSON: {"sql": "...", "explanation": "..."}'
    )

    llm = _llm().bind_tools(SQL_WRITER_TOOLS)
    response = llm.invoke(
        [SystemMessage(content=SQL_WRITER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(response.content) or {}
    sql = parsed.get("sql", "")

    updated_plan = [
        s.model_copy(update={"status": "running"})
        if s.id == step.id else s for s in state.plan
    ]
    return {"last_sql": sql, "plan": updated_plan, "verification_feedback": "", "current_step": step}


# ---------------------------------------------------------------------------
# Node: execute_sql
# ---------------------------------------------------------------------------

def execute_sql(state: AnalyticsState) -> dict:
    result = run_sql.invoke({"query": state.last_sql})
    if result.startswith("ERROR"):
        updated_plan = [
            s.model_copy(update={"status": "failed", "result": result})
            if state.current_step and s.id == state.current_step.id else s
            for s in state.plan
        ]
        return {"last_query_result": result, "last_query_metadata": {},
                "plan": updated_plan, "error": result}
    parsed = json.loads(result)
    return {
        "last_query_result": json.dumps(parsed.get("rows", []), default=str),
        "last_query_metadata": parsed.get("metadata", {}),
        "error": "",
    }


# ---------------------------------------------------------------------------
# Node: verifier — NO tools; reasons purely on state data
# ---------------------------------------------------------------------------

def verifier(state: AnalyticsState) -> dict:
    step = state.current_step
    row_count = state.last_query_metadata.get("row_count", "unknown")
    table_preview = _rows_to_markdown(state.last_query_result)
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
    new_status = "done" if verdict in ("pass", "warning") else "failed"
    updated_plan = [
        s.model_copy(update={"status": new_status, "result": feedback})
        if step and s.id == step.id else s for s in state.plan
    ]
    updates: dict = {
        "verification_verdict": verdict,
        "verification_feedback": feedback,
        "plan": updated_plan,
        "current_step": None,
    }
    if corrected_sql:
        updates["last_sql"] = str(corrected_sql)
    return updates
