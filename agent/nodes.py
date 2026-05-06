"""LangGraph node functions – one per agent role."""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts import (
    ORCHESTRATOR_SYSTEM,
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
    """Safely extract a plain string from a message content.

    LangGraph Studio sends content as a list of block dicts:
        [{'type': 'text', 'text': 'What do you understand?'}]
    Plain LLM responses send content as a str.
    This helper handles both.
    """
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


def _parse_json_from_response(text: str) -> dict:
    """Extract the first JSON object from an LLM text response."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Node: orchestrator
# ---------------------------------------------------------------------------

def orchestrator(state: AnalyticsState) -> dict:
    """Plan the query or synthesise the final answer when all steps are done."""
    llm = _llm().bind_tools(ORCHESTRATOR_TOOLS)

    # Resolve user_query from state or latest HumanMessage (handles Studio blocks)
    user_query = state.user_query
    if not user_query:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                user_query = _extract_text(msg.content)
                break

    # If all plan steps are done, synthesise the final answer
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
            [SystemMessage(content=ORCHESTRATOR_SYSTEM), HumanMessage(content=synthesis_prompt)]
        )
        parsed = _parse_json_from_response(response.content)
        final = parsed.get("final_answer", response.content)
        return {
            "final_answer": final,
            "messages": [AIMessage(content=final)],
        }

    # Build the execution plan
    plan_prompt = (
        f"User question: {user_query}\n\n"
        "Create a minimal execution plan as JSON. "
        'Format: {"plan": [{"id": 1, "type": "sql", "description": "..."}]}'
    )
    response = llm.invoke(
        [SystemMessage(content=ORCHESTRATOR_SYSTEM), HumanMessage(content=plan_prompt)]
    )
    parsed = _parse_json_from_response(response.content)
    plan = [PlanStep(**step) for step in parsed.get("plan", [])]
    return {"user_query": user_query, "plan": plan}


# ---------------------------------------------------------------------------
# Node: profiler
# ---------------------------------------------------------------------------

def profiler(state: AnalyticsState) -> dict:
    """Run data profiling for the current plan step."""
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
    """Generate DuckDB SQL for the current plan step."""
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
    parsed = _parse_json_from_response(response.content)
    sql = parsed.get("sql", "")

    updated_plan = [
        s.model_copy(update={"status": "running"})
        if s.id == step.id else s
        for s in state.plan
    ]
    return {"last_sql": sql, "plan": updated_plan, "verification_feedback": ""}


# ---------------------------------------------------------------------------
# Node: execute_sql  (deterministic – no LLM)
# ---------------------------------------------------------------------------

def execute_sql(state: AnalyticsState) -> dict:
    """Execute the SQL produced by sql_writer and store results in state."""
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
    """Cross-check the SQL result and return a pass/fail verdict."""
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
    parsed = _parse_json_from_response(response.content)
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
