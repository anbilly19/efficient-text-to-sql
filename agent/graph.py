"""LangGraph StateGraph assembly.

Intent routing:
  load      → load_file_node → END
  analytics → profiler | sql_writer → execute_sql → verifier → [next step | orchestrator synthesis] → END
  chitchat  → END

Multi-step plans advance directly through verifier → next node WITHOUT re-entering
orthestrator mid-plan. Orchestrator is only re-entered for final synthesis (all steps done)
or on a verifier fail (retry the same step via sql_writer).
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
    """Advance the plan without returning to orchestrator mid-plan.

    - verifier FAIL  → retry the same step via sql_writer (corrected SQL already in state)
    - more pending steps exist  → go directly to profiler or sql_writer
    - all steps done  → orchestrator (synthesis only)
    """
    # Retry on fail
    if state.verification_verdict == "fail":
        return "sql_writer"

    # Advance to the next pending step directly—skip orchestrator
    next_step = _next_pending_step(state)
    if next_step is not None:
        return "profiler" if next_step.type == "profile" else "sql_writer"

    # All done → orchestrator for synthesis
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
        "profiler": "profiler",
        "sql_writer": "sql_writer",
        END: END,
    },
)

builder.add_edge("load_file_node", END)
builder.add_edge("profiler",       "sql_writer")   # profiler feeds directly into sql_writer
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
