"""LangGraph node functions – one per agent role."""
from __future__ import annotations

import json
import os
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


def _parse_json_from_response(text: str) -> dict:
    result = _try_parse_json(text)
    if result is None:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    return result


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
            return "No tables loaded yet. Ask the user to load a file first."
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


# ---------------------------------------------------------------------------
# Node: orchestrator
# ---------------------------------------------------------------------------

def orchestrator(state: AnalyticsState) -> dict:
    """Plan the query or synthesise the final answer when all steps are done."""

    # Resolve user_query (handles Studio content-block lists)
    user_query = state.user_query
    if not user_query:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                user_query = _extract_text(msg.content)
                break

    # Fetch available tables to inject into every prompt
    tables_context = _get_available_tables()

    ORCHESTRATOR_SYSTEM = f"""You are the Orchestrator of a DuckDB analytics agent.

Available tables in DuckDB:
{tables_context}

If the user's message is a greeting, small talk, or NOT a data/analytics question
(e.g. "hi", "hello", "how are you"), respond ONLY with:
{{"final_answer": "<friendly reply mentioning what tables are available>"}}

Otherwise, for any data or analytics question, create a minimal execution plan:
{{"plan": [{{"id": 1, "type": "sql", "description": "..."}}]}}

Step types:
- "profile" → explore a column's data distribution
- "sql"     → generate a DuckDB SELECT query

Rules:
- Never generate Python code.
- Keep plans minimal: profile only when column semantics are ambiguous.
- When re-planning after a failure, include the verifier's feedback in the new step description.
- Always reference the correct table name from the available tables listed above.
"""

    # ---- Synthesis: all steps finished, compose final answer ----
    if state.plan and all(s.status in ("done", "failed") for s in state.plan):
        synthesis_prompt = (
            f"User question: {user_query}\n\n"
            f"Query result (first 50 rows): {state.last_query_result}\n\n"
            f"Verification verdict: {state.verification_verdict}\n"
            f"Verification feedback: {state.verification_feedback}\n\n"
            "Synthesise a clear, concise final answer for the user. "
            'Output JSON: {"final_answer": "..."}'
        )
        response = _llm().invoke(
            [SystemMessage(content=ORCHESTRATOR_SYSTEM),
             HumanMessage(content=synthesis_prompt)]
        )
        parsed = _try_parse_json(response.content) or {}
        final = parsed.get("final_answer", response.content)
        return {
            "final_answer": final,
            "messages": [AIMessage(content=final)],
        }

    # ---- Planning: build execution plan (or reply to chitchat) ----
    plan_prompt = (
        f"Available tables:\n{tables_context}\n\n"
        f"User message: {user_query}\n\n"
        "If this is a data/analytics question, output a JSON plan using the correct table names above. "
        "If it is chitchat or a greeting, output a friendly JSON final_answer mentioning the available tables."
    )
    llm = _llm().bind_tools(ORCHESTRATOR_TOOLS)
    response = llm.invoke(
        [SystemMessage(content=ORCHESTRATOR_SYSTEM),
         HumanMessage(content=plan_prompt)]
    )

    parsed = _try_parse_json(response.content)

    # LLM returned no JSON at all (pure conversational reply)
    if parsed is None:
        reply = response.content or "Hi! I'm a DuckDB analytics agent. Ask me a data question!"
        return {
            "final_answer": reply,
            "user_query": user_query,
            "messages": [AIMessage(content=reply)],
        }

    # LLM returned a final_answer directly (chitchat detected)
    if "final_answer" in parsed and "plan" not in parsed:
        reply = parsed["final_answer"]
        return {
            "final_answer": reply,
            "user_query": user_query,
            "messages": [AIMessage(content=reply)],
        }

    # Normal analytics plan
    plan = [PlanStep(**step) for step in parsed.get("plan", [])]
    return {"user_query": user_query, "plan": plan}


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
