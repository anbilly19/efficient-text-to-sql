"""LangGraph StateGraph assembly.

Intent routing:
  load      → load_file_node → END
  analytics → profiler | sql_writer → execute_sql → verifier → [next step | orchestrator synthesis] → END
  chitchat  → END

Verifier NEVER calls tools — it reasons on state data only.
Multi-step plans advance directly: verifier → next pending step (no orchestrator mid-plan).
Orchestrator re-entered ONLY for final synthesis.
Hard cap: after MAX_RETRIES consecutive verifier fails, force step to done and continue.
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    execute_sql,
    load_file_node,
    orchestrator,
    profiler,
    sql_writer,
    verifier,
)
from agent.state import AnalyticsState, PlanStep

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Router helpers
# ---------------------------------------------------------------------------

def _next_pending_step(state: AnalyticsState) -> PlanStep | None:
    return next((s for s in state.plan if s.status == "pending"), None)


def _all_steps_done(state: AnalyticsState) -> bool:
    return bool(state.plan) and all(s.status in ("done", "failed") for s in state.plan)


def route_after_orchestrator(
    state: AnalyticsState,
) -> Literal["load_file_node", "profiler", "sql_writer", "__end__"]:
    if state.load_file_path:
        return "load_file_node"
    step = _next_pending_step(state)
    if step is not None:
        return "profiler" if step.type == "profile" else "sql_writer"
    if state.final_answer:
        return END
    return END


def route_after_verifier(
    state: AnalyticsState,
) -> Literal["orchestrator", "profiler", "sql_writer"]:
    """Advance plan without re-entering orchestrator mid-plan.

    - fail + retries remaining  → sql_writer (retry)
    - fail + retries exhausted  → mark step done, fall through
    - more pending steps        → directly to profiler/sql_writer
    - all steps done            → orchestrator (synthesis)
    """
    if state.verification_verdict == "fail":
        if state.retry_count < MAX_RETRIES:
            return "sql_writer"
        # Retries exhausted — treat current step as done and advance
        # (step was already marked failed in verifier node; just move on)

    next_step = _next_pending_step(state)
    if next_step is not None:
        return "profiler" if next_step.type == "profile" else "sql_writer"

    return "orchestrator"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

builder = StateGraph(AnalyticsState)

builder.add_node("orchestrator",   orchestrator)
builder.add_node("load_file_node", load_file_node)
builder.add_node("profiler",       profiler)
builder.add_node("sql_writer",     sql_writer)
builder.add_node("execute_sql",    execute_sql)
builder.add_node("verifier",       verifier)

builder.add_edge(START, "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {
        "load_file_node": "load_file_node",
        "profiler":       "profiler",
        "sql_writer":     "sql_writer",
        END:              END,
    },
)

builder.add_edge("load_file_node", END)
builder.add_edge("profiler",       "sql_writer")
builder.add_edge("sql_writer",     "execute_sql")
builder.add_edge("execute_sql",    "verifier")

builder.add_conditional_edges(
    "verifier",
    route_after_verifier,
    {
        "orchestrator": "orchestrator",
        "profiler":     "profiler",
        "sql_writer":   "sql_writer",
    },
)

graph = builder.compile()
graph.name = "DuckDB Analytics Agent"
