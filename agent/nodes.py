"""LangGraph node functions – one per agent role."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AnalyticsState, PlanStep
from agent.tools import (
    get_schema,
    load_file,
    profile_column,
    run_sql,
    run_test_query,
    lookup_semantic,
)
from agent.database import get_connection, get_semantic_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=_MODEL, temperature=0)


def _try_parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _extract_text(content: Any) -> str:
    """Extract plain text from an LLM response content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _get_most_recent_table() -> str | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT dataset_name FROM _schema_catalog ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _get_all_tables() -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT dataset_name FROM _schema_catalog ORDER BY dataset_name"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _get_schema_for_table(table_name: str) -> str:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT column_name, data_type, COALESCE(description, '')
            FROM _schema_catalog
            WHERE dataset_name = ?
            ORDER BY rowid
            """,
            [table_name],
        ).fetchall()
        if not rows:
            return f"(no schema found for {table_name!r})"
        lines = [f"Table: {table_name}", "-" * 40]
        for col, dtype, desc in rows:
            lines.append(f"  {col} ({dtype})" + (f" — {desc}" if desc else ""))
        return "\n".join(lines)
    except Exception as e:
        return f"(error reading schema: {e})"


def _extract_table_from_plan(plan: list[PlanStep], query: str) -> str | None:
    """Guess the target table from the plan descriptions or fall back to most recent."""
    for step in plan:
        if step.description:
            conn = get_connection()
            try:
                tables = [
                    r[0]
                    for r in conn.execute(
                        "SELECT DISTINCT dataset_name FROM _schema_catalog"
                    ).fetchall()
                ]
                for t in tables:
                    if t.lower() in step.description.lower():
                        return t
            except Exception:
                pass
    return _get_most_recent_table()


def _get_columns_for_table(table_name: str) -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT column_name FROM _schema_catalog WHERE dataset_name = ? ORDER BY rowid",
            [table_name],
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _rows_to_markdown(result: str | None) -> str:
    """Convert a pipe-delimited result string to a markdown table (best-effort)."""
    if not result or result.startswith("ERROR"):
        return result or ""
    lines = [l for l in result.strip().split("\n") if l.strip()]
    if not lines:
        return result
    if "|" in lines[0]:
        return result
    return result


def _sanitize_sql(sql: str, table_name: str | None) -> str:
    """Normalise SQL returned by the LLM."""
    if not sql:
        return sql

    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(lines[1:-1]).strip()

    sql = sql.rstrip(";").strip()
    sql = re.sub(r'"([^"]+)""', r'"\1"', sql)
    sql = re.sub(r'""([^"]+)"', r'"\1"', sql)

    if not table_name:
        return sql

    columns = _get_columns_for_table(table_name)
    for col in columns:
        if re.search(r'[^a-zA-Z0-9_]', col):
            col_pattern = rf'(?:"?\w+"?\.)?\"?{re.escape(col)}\"?'
            sql = re.sub(col_pattern, f'"{col}"', sql)

    unquoted = re.compile(rf'\b{re.escape(table_name)}\b(?!")', re.IGNORECASE)
    sql = unquoted.sub(f'"{table_name}"', sql)

    return sql


# ---------------------------------------------------------------------------
# Node: load_file_node  (handles "load …" commands)
# ---------------------------------------------------------------------------

def load_file_node(state: AnalyticsState) -> dict:
    """Detect and execute file-load commands, then reset conversation state."""
    query = (state.user_query or "").strip()

    match = re.match(
        r"(?:load\s+(?:file\s+(?:at\s+)?)?)"
        r"([^\s]+(?:\s+[^\s]+)*?)\s+as\s+(\w+)",
        query,
        re.IGNORECASE,
    )
    if not match:
        return {}

    path_str, dataset_name = match.group(1).strip(), match.group(2).strip()

    path = Path(path_str)
    if not path.exists():
        for candidate in [Path("data") / path_str, Path(".") / path_str]:
            if candidate.exists():
                path = candidate
                break

    try:
        result = load_file.invoke({"path": str(path), "dataset_name": dataset_name})
        reply = f"\u2705 Successfully loaded **{dataset_name}** from `{path}`. You can now ask questions about it."
    except Exception as exc:
        reply = f"\u274c Failed to load `{path_str}`: {exc}"

    base_reset: dict = {
        "plan": [],
        "current_step": None,
        "last_sql": "",
        "last_query_result": "",
        "last_query_metadata": {},
        "verification_verdict": "",
        "verification_feedback": "",
        "retry_count": 0,
        "error": "",
        "final_answer": reply,
    }
    return {**base_reset, "messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# Node: orchestrator
# ---------------------------------------------------------------------------

def orchestrator(state: AnalyticsState) -> dict:
    """Classify intent and build a plan, or synthesise the final answer."""

    base_reset: dict = {
        "plan": state.plan,
        "current_step": None,
        "last_sql": "",
        "last_query_result": "",
        "last_query_metadata": {},
        "verification_verdict": "",
        "verification_feedback": "",
        "retry_count": 0,
        "error": "",
        "final_answer": "",
    }

    # ── Synthesis: fire when ALL steps are done or failed ─────────────────────
    if state.plan and all(s.status in ("done", "failed") for s in state.plan):
        completed = [s for s in state.plan if s.status == "done" and s.result]
        if not completed:
            answer = "I couldn't retrieve any results. Please try rephrasing your question."
            return {**base_reset, "final_answer": answer,
                    "messages": [AIMessage(content=answer)]}

        results_block = "\n\n".join(
            f"**Step {s.id} — {s.description}**\n{s.result}" for s in completed
        )
        synthesis_prompt = (
            f"Original question: {state.user_query}\n\n"
            f"Query results:\n{results_block}\n\n"
            "Write a clear, concise answer to the original question using the results above. "
            "Be specific — include numbers, names, and values from the data. "
            "Do not mention SQL, steps, or technical details."
        )
        response = _llm().invoke(
            [SystemMessage(content="You are a helpful data analyst."),
             HumanMessage(content=synthesis_prompt)]
        )
        answer = _extract_text(response.content)
        return {**base_reset, "final_answer": answer,
                "messages": [AIMessage(content=answer)]}

    # Already have a plan in progress — don't re-plan
    if state.plan and any(s.status == "pending" for s in state.plan):
        return {}

    current_query = state.user_query or ""
    all_tables = _get_all_tables()
    most_recent_table = _get_most_recent_table()
    tables_context = "\n".join(
        f"- {t}\n{_get_schema_for_table(t)}" for t in all_tables
    ) or "(no tables loaded)"

    # ── Fast-path: schema / column questions ────────────────────────────
    schema_keywords = re.compile(
        r"\b(schema|columns?|fields?|structure|what.+table|show.+table|describe)\b",
        re.IGNORECASE,
    )
    target_table: str | None = None
    if schema_keywords.search(current_query):
        for t in all_tables:
            if t.lower() in current_query.lower():
                target_table = t
                break
        if not target_table and most_recent_table:
            target_table = most_recent_table

        if not all_tables:
            reply = "No tables are loaded yet. Load a file first with: `load file at <path> as <name>`."
        elif target_table:
            schema_text = _get_schema_for_table(target_table)
            semantic_text = get_semantic_context(target_table)
            reply = (
                f"Here is the schema for the **{target_table}** table:\n\n"
                f"```\n{schema_text}\n```\n\n"
                f"{semantic_text}"
            )
        else:
            reply = f"Here are all loaded tables:\n\n```\n{tables_context}\n```"

        return {**base_reset, "final_answer": reply, "messages": [AIMessage(content=reply)]}

    # ── LLM planning ──────────────────────────────────────────────────────
    semantic_context = get_semantic_context(most_recent_table)

    ORCHESTRATOR_SYSTEM = f"""You are the Orchestrator of a DuckDB analytics agent.

Available tables in DuckDB:
{tables_context}

Most recently loaded table: {most_recent_table or 'none'}

---

{semantic_context}

Classify the user message into exactly ONE intent:

1. ANALYTICS – use when the question asks about data IN a loaded table.
   {{"intent": "analytics", "plan": [{{"id": 1, "type": "sql", "description": "..."}}]}}
   Step types: "profile" (explore a column) or "sql" (SELECT query).
   Every step description MUST name the exact table.
   Default table if unspecified: "{most_recent_table}".

2. CHITCHAT – use for EVERYTHING else.
   {{"intent": "chitchat", "final_answer": "<helpful reply>"}}

Decision rule: if the question references a loaded table or asks about data/columns, pick ANALYTICS.
Output ONLY the JSON. No markdown, no explanation.
"""

    response = _llm().invoke(
        [SystemMessage(content=ORCHESTRATOR_SYSTEM),
         HumanMessage(content=current_query)]
    )
    raw_text = _extract_text(response.content)
    parsed = _try_parse_json(raw_text)

    if parsed is None and most_recent_table:
        parsed = {
            "intent": "analytics",
            "plan": [{"id": 1, "type": "sql",
                      "description": f"Answer the user question against the {most_recent_table} table: {current_query}"}],
        }

    if parsed is None:
        answer = raw_text or "Sorry, I could not understand that request."
        return {**base_reset, "final_answer": answer,
                "messages": [AIMessage(content=answer)]}

    intent = parsed.get("intent", "analytics")

    if intent == "chitchat":
        answer = parsed.get("final_answer", "")
        return {**base_reset, "final_answer": answer,
                "messages": [AIMessage(content=answer)]}

    raw_steps = parsed.get("plan", [])

    if not raw_steps and most_recent_table:
        raw_steps = [{"id": 1, "type": "sql",
                      "description": f"Answer the user question against the {most_recent_table} table: {current_query}"}]

    plan = [
        PlanStep(
            id=s.get("id", i + 1),
            type=s.get("type", "sql"),
            description=s.get("description", ""),
            status="pending",
        )
        for i, s in enumerate(raw_steps)
    ]

    return {**base_reset, "plan": plan}


# ---------------------------------------------------------------------------
# Node: set_step  (router — picks the next pending step)
# ---------------------------------------------------------------------------

def set_step(state: AnalyticsState) -> dict:
    for step in state.plan:
        if step.status == "pending":
            return {"current_step": step}
    return {"current_step": None}


# ---------------------------------------------------------------------------
# Node: profiler
# ---------------------------------------------------------------------------

def profiler(state: AnalyticsState) -> dict:
    step = state.current_step
    if step is None:
        return {}

    table_name = _extract_table_from_plan(state.plan, state.user_query or "")
    if not table_name:
        updated = [
            s.model_copy(update={"status": "failed", "result": "no table found"})
            if s.id == step.id else s
            for s in state.plan
        ]
        return {"plan": updated}

    columns = _get_columns_for_table(table_name)
    if not columns:
        updated = [
            s.model_copy(update={"status": "failed", "result": "no columns found"})
            if s.id == step.id else s
            for s in state.plan
        ]
        return {"plan": updated}

    target_col = None
    for col in columns:
        if col.lower() in (step.description or "").lower():
            target_col = col
            break
    if not target_col:
        target_col = columns[0]

    try:
        result = profile_column.invoke(
            {"dataset": table_name, "column": target_col}
        )
    except Exception as exc:
        result = f"Profile failed: {exc}"

    updated = [
        s.model_copy(update={"status": "done", "result": str(result)})
        if s.id == step.id else s
        for s in state.plan
    ]
    return {"plan": updated, "current_step": step.model_copy(update={"status": "done", "result": str(result)})}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SQL_WRITER_SYSTEM = """You are a DuckDB SQL expert. Write precise, read-only SELECT queries.

Rules:
- Output ONLY a JSON object: {"sql": "<query>"}
- Always double-quote identifiers with spaces or special characters: "My Column"
- Use DuckDB syntax: TRY_CAST, STRFTIME, REGEXP_MATCHES, etc.
- Never use INSERT, UPDATE, DELETE, DROP, CREATE, or any DDL/DML.
- If a column name looks like a date stored as VARCHAR, wrap it:
  TRY_CAST("Date Column" AS DATE)
- Return at most 1000 rows unless the question asks for all.
- Do not call any tools. Output the JSON only.
"""

VERIFIER_SYSTEM = """You are a SQL result verifier for a DuckDB analytics agent.

Your job:
- Check whether the SQL correctly answers the sub-task.
- If the result is clearly wrong or the SQL has a bug, provide a corrected SQL.
- Output ONLY a JSON object.

Output format:
{"verdict": "pass|fail|warning", "feedback": "...", "corrected_sql": "(only if fail)"}

Rules:
- verdict=pass  → result looks correct
- verdict=warning → result is plausible but uncertain (treated as pass by the agent)
- verdict=fail  → clear bug; you MUST provide corrected_sql
- Do NOT call any tools.
"""


# ---------------------------------------------------------------------------
# Node: sql_writer
# ---------------------------------------------------------------------------

def sql_writer(state: AnalyticsState) -> dict:
    step = state.current_step
    if step is None:
        return {}

    table_name = _extract_table_from_plan(state.plan, state.user_query or "")
    if not table_name:
        table_name = _get_most_recent_table()

    schema_text = _get_schema_for_table(table_name) if table_name else "(no schema)"

    def _updated_plan(status: str, result: str) -> list[PlanStep]:
        return [
            s.model_copy(update={"status": status, "result": result})
            if s.id == step.id else s
            for s in state.plan
        ]

    # On retry, if verifier supplied a corrected SQL, use it directly
    if state.retry_count > 0 and state.last_sql:
        sql = _sanitize_sql(state.last_sql, table_name)
        updated_plan = _updated_plan("pending", "")
        return {"last_sql": sql, "plan": updated_plan, "current_step": step}

    prior_step_notes = "\n".join(
        f"- Step {s.id} ({s.description}): {s.result}"
        for s in state.plan
        if s.status == "done" and s.result and s.id != step.id
    )

    prompt = (
        f"Schema:\n{schema_text}\n\n"
        + (f"Prior step results:\n{prior_step_notes}\n\n" if prior_step_notes else "")
        + f"Task: {step.description}\n\n"
        "Output ONLY: {\"sql\": \"<your SELECT query>\"}"
    )

    response = _llm().invoke(
        [SystemMessage(content=SQL_WRITER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(_extract_text(response.content)) or {}
    sql = parsed.get("sql", "")

    # Fallback: LLM returned a tool call instead of plain JSON
    if not sql and hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            args = tc.get("args") or {}
            sql = args.get("query") or args.get("sql") or ""
            if sql:
                break

    if not sql or not sql.strip():
        updated_plan = _updated_plan("failed", "sql_writer produced empty SQL")
        return {
            "last_sql": "",
            "plan": updated_plan,
            "error": "sql_writer produced empty SQL",
            "current_step": step,
        }

    sql = _sanitize_sql(sql, table_name)
    updated_plan = _updated_plan("pending", "")
    return {"last_sql": sql, "plan": updated_plan, "current_step": step}


# ---------------------------------------------------------------------------
# Node: execute_sql
# ---------------------------------------------------------------------------

def execute_sql(state: AnalyticsState) -> dict:
    step = state.current_step

    if not state.last_sql or not state.last_sql.strip():
        def _fail_plan() -> list[PlanStep]:
            return [
                s.model_copy(update={"status": "failed", "result": "empty SQL"})
                if state.current_step and s.id == state.current_step.id else s
                for s in state.plan
            ]
        return {
            "last_query_result": "ERROR: empty SQL",
            "plan": _fail_plan(),
            "last_query_metadata": {},
        }

    try:
        result = run_sql.invoke({"query": state.last_sql})
        if isinstance(result, dict):
            rows = result.get("rows", "")
            metadata = {k: v for k, v in result.items() if k != "rows"}
            return {
                "last_query_result": str(rows),
                "last_query_metadata": metadata,
            }
        return {"last_query_result": str(result), "last_query_metadata": {}}
    except Exception as exc:
        error_msg = f"ERROR: {exc}"
        updated_plan = [
            s.model_copy(update={"status": "failed", "result": error_msg})
            if step and s.id == step.id else s
            for s in state.plan
        ]
        return {
            "last_query_result": error_msg,
            "plan": updated_plan,
            "last_query_metadata": {},
        }


# ---------------------------------------------------------------------------
# Node: verifier
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
    corrected_sql = str(parsed.get("corrected_sql", ""))

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
