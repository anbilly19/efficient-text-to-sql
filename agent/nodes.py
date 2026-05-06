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
    VERIFIER_TOOLS,
    load_file,
    run_sql,
)
from agent.database import get_connection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _extract_text(content: Any) -> str:
    """Safely extract a plain string from a message content (handles Studio block lists)."""
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
    """Try to extract a JSON object from text. Returns None if not found."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _get_available_tables() -> str:
    """Query DuckDB for all user-loaded tables and return a summary string."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT dataset_name, column_name, data_type
            FROM _schema_catalog
            ORDER BY dataset_name, column_name
            """
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


_LOAD_PATTERNS = [
    re.compile(
        r"load\s+(?:file\s+)?at\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"import\s+(?P<path>\S+)\s+as\s+(?P<dataset>\w+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"load\s+(?P<path>\S+\.(?:xlsx|xls|csv|parquet))\s+as\s+(?P<dataset>\w+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"load\s+(?:file\s+)?(?P<path>\S+\.(?:xlsx|xls|csv|parquet))",
        re.IGNORECASE,
    ),
]


def _detect_load_intent(text: str) -> dict | None:
    """Return {path, dataset} if the message is clearly a file-load request, else None."""
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


# ---------------------------------------------------------------------------
# Node: orchestrator
# ---------------------------------------------------------------------------

def orchestrator(state: AnalyticsState) -> dict:
    """Classify user intent: load / analytics / chitchat / synthesise."""

    # Always read from the latest HumanMessage so each turn is independent
    current_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            current_query = _extract_text(msg.content)
            break

    tables_context = _get_available_tables()

    # ── Reset ALL stale fields at the top of every new turn ─────────────────
    # None is used for optional fields so LangGraph sees a real value change.
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
        "load_file_path": None,    # None = not a load turn
        "load_file_dataset": None,
        "error": "",
    }

    # ── Synthesis: all plan steps completed ──────────────────────────────
    if state.plan and all(s.status in ("done", "failed") for s in state.plan):
        synthesis_prompt = (
            f"User question: {state.user_query}\n\n"
            f"Query result (first 50 rows): {state.last_query_result}\n\n"
            f"Verification verdict: {state.verification_verdict}\n"
            f"Verification feedback: {state.verification_feedback}\n\n"
            "Synthesise a clear, concise final answer for the user. "
            'Output JSON: {"final_answer": "..."}'
        )
        response = _llm().invoke([HumanMessage(content=synthesis_prompt)])
        parsed = _try_parse_json(response.content) or {}
        final = parsed.get("final_answer", response.content)
        return {
            **base_reset,         # clears plan so next turn starts fresh
            "final_answer": final,
            "messages": [AIMessage(content=final)],
        }

    # ── Fast-path: regex load detection (bypasses LLM entirely) ────────────
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

Classify the user message into exactly ONE intent and respond with matching JSON:

1. ANALYTICS – data or SQL question:
   {{"intent": "analytics", "plan": [{{"id": 1, "type": "sql", "description": "..."}}]}}
   Step types: "profile" (explore column) or "sql" (SELECT query).
   Always use the correct table name from the available tables above.

2. CHITCHAT – greeting, small talk, or off-topic:
   {{"intent": "chitchat", "final_answer": "<friendly reply mentioning available tables and how to load files>"}}

Rules:
- Output ONLY the JSON. No markdown, no explanation.
- Never generate Python code.
- Keep plans minimal (1-2 steps).
"""

    # ── LLM classification (analytics vs chitchat only) ──────────────────
    llm = _llm().bind_tools(ORCHESTRATOR_TOOLS)
    response = llm.invoke(
        [SystemMessage(content=ORCHESTRATOR_SYSTEM),
         HumanMessage(content=current_query)]
    )

    parsed = _try_parse_json(response.content)

    if parsed is None:
        reply = response.content or "Hi! Ask a data question or say: load file at <path> as <name>."
        return {
            **base_reset,
            "final_answer": reply,
            "messages": [AIMessage(content=reply)],
        }

    intent = parsed.get("intent", "chitchat")

    if intent == "analytics":
        plan = [PlanStep(**step) for step in parsed.get("plan", [])]
        return {**base_reset, "plan": plan}

    # chitchat
    reply = parsed.get("final_answer", "Hi! Ask a data question or load a file.")
    return {
        **base_reset,
        "final_answer": reply,
        "messages": [AIMessage(content=reply)],
    }


# ---------------------------------------------------------------------------
# Node: load_file_node
# ---------------------------------------------------------------------------

def load_file_node(state: AnalyticsState) -> dict:
    """Load a file into DuckDB using the path and dataset name set by orchestrator."""
    path = state.load_file_path
    dataset = state.load_file_dataset

    if not path:
        reply = "I couldn't find a file path. Please say: load file at <path> as <name>."
        return {
            "final_answer": reply,
            "messages": [AIMessage(content=reply)],
            "load_file_path": None,
            "load_file_dataset": None,
        }

    if not dataset:
        dataset = Path(path).stem.replace(" ", "_").replace("-", "_").lower()

    result = load_file.invoke({"path": path, "dataset_name": dataset})

    if result.startswith("ERROR"):
        reply = f"❌ Failed to load file: {result}"
    else:
        reply = f"✅ {result}\n\nYou can now ask questions about the `{dataset}` table!"

    return {
        "final_answer": reply,
        "messages": [AIMessage(content=reply)],
        "load_file_path": None,
        "load_file_dataset": None,
    }


# ---------------------------------------------------------------------------
# Node: profiler
# ---------------------------------------------------------------------------

def profiler(state: AnalyticsState) -> dict:
    llm = _llm().bind_tools(PROFILER_TOOLS)
    step = state.current_step

    response = llm.invoke([
        SystemMessage(content=PROFILER_SYSTEM),
        HumanMessage(content=f"Profiling task: {step.description}\nUser question: {state.user_query}"),
    ])
    result_text = response.content or "No profiling result."

    updated_plan = [
        s.model_copy(update={"status": "done", "result": result_text})
        if s.id == step.id else s
        for s in state.plan
    ]
    return {"plan": updated_plan, "current_step": None}


# ---------------------------------------------------------------------------
# Node: sql_writer
# ---------------------------------------------------------------------------

def sql_writer(state: AnalyticsState) -> dict:
    llm = _llm().bind_tools(SQL_WRITER_TOOLS)
    step = state.current_step

    profiling_notes = "\n".join(
        f"- {s.description}: {s.result}"
        for s in state.plan
        if s.type == "profile" and s.status == "done" and s.result
    )

    prompt = (
        f"Sub-task: {step.description}\n"
        f"User question: {state.user_query}\n"
        f"Profiling notes:\n{profiling_notes or 'None'}\n\n"
        f"Verifier feedback (if retry): {state.verification_feedback or 'None'}\n\n"
        'Output JSON: {"sql": "...", "explanation": "..."}'
    )

    response = llm.invoke(
        [SystemMessage(content=SQL_WRITER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(response.content) or {}
    sql = parsed.get("sql", "")

    updated_plan = [
        s.model_copy(update={"status": "running"})
        if s.id == step.id else s
        for s in state.plan
    ]
    return {"last_sql": sql, "plan": updated_plan, "verification_feedback": ""}


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
        return {
            "last_query_result": result,
            "last_query_metadata": {},
            "plan": updated_plan,
            "error": result,
        }
    parsed = json.loads(result)
    return {
        "last_query_result": json.dumps(parsed.get("rows", []), default=str),
        "last_query_metadata": parsed.get("metadata", {}),
        "error": "",
    }


# ---------------------------------------------------------------------------
# Node: verifier
# ---------------------------------------------------------------------------

def verifier(state: AnalyticsState) -> dict:
    llm = _llm().bind_tools(VERIFIER_TOOLS)
    step = state.current_step

    prompt = (
        f"Sub-task: {step.description if step else 'unknown'}\n"
        f"SQL executed:\n{state.last_sql}\n\n"
        f"Result metadata: {json.dumps(state.last_query_metadata)}\n"
        f"First rows: {state.last_query_result[:2000]}\n\n"
        'Output JSON: {"verdict": "pass|fail|warning", "feedback": "...", "corrected_sql": "(only if fail)"}'
    )

    response = llm.invoke(
        [SystemMessage(content=VERIFIER_SYSTEM), HumanMessage(content=prompt)]
    )
    parsed = _try_parse_json(response.content) or {"verdict": "pass", "feedback": ""}
    verdict = parsed.get("verdict", "pass")
    feedback = parsed.get("feedback", "")
    corrected_sql = parsed.get("corrected_sql", "")

    new_status = "done" if verdict in ("pass", "warning") else "failed"
    updated_plan = [
        s.model_copy(update={"status": new_status, "result": feedback})
        if step and s.id == step.id else s
        for s in state.plan
    ]

    updates: dict = {
        "verification_verdict": verdict,
        "verification_feedback": feedback,
        "plan": updated_plan,
        "current_step": None,
    }
    if corrected_sql:
        updates["last_sql"] = corrected_sql
    return updates
