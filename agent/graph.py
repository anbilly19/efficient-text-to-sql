"""LangGraph StateGraph assembly.

Routing:
  chitchat/load  -> END
  simple sql     -> sql_writer -> execute_sql -> verifier -> orchestrator -> END
  with profile   -> profiler -> sql_writer -> execute_sql -> verifier -> orchestrator -> END
  on fail        -> verifier -> sql_writer (max 2 retries) -> orchestrator -> END
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    executor,
    load_file_node,
    orchestrator,
    profiler,
    sql_writer,
    verifier,
)
from agent.state import AnalyticsState, PlanStep

MAX_RETRIES = 2
MAX_PLAN_STEPS = 5  # hard cap to prevent infinite re-planning


def _next_pending_step(state: AnalyticsState) -> PlanStep | None:
    return next((s for s in state.plan if s.status == "pending"), None)


def route_after_orchestrator(
    state: AnalyticsState,
) -> Literal["load_file_node", "profiler", "sql_writer", "__end__"]:
    if state.load_file_path:
        return "load_file_node"
    # Hard cap: bail out if plan grew too large (re-planning loop)
    if len(state.plan) > MAX_PLAN_STEPS:
        return END
    step = _next_pending_step(state)
    if step is not None:
        return "profiler" if step.type == "profile" else "sql_writer"
    return END


def route_after_execute(
    state: AnalyticsState,
) -> Literal["verifier", "orchestrator"]:
    """Always route through verifier after execution.

    The only exception is a hard SQL failure with no result to verify,
    in which case we go straight to orchestrator to surface the error.
    """
    if state.error and not state.last_query_result:
        return "orchestrator"
    return "verifier"


def route_after_verifier(
    state: AnalyticsState,
) -> Literal["orchestrator", "sql_writer"]:
    # Hard stop: always go to orchestrator after MAX_RETRIES
    if state.retry_count >= MAX_RETRIES:
        return "orchestrator"
    if state.verification_verdict == "fail":
        return "sql_writer"
    next_step = _next_pending_step(state)
    if next_step is not None:
        return "sql_writer"
    return "orchestrator"


# ---------------------------------------------------------------------------
builder = StateGraph(AnalyticsState)

builder.add_node("orchestrator",   orchestrator)
builder.add_node("load_file_node", load_file_node)
builder.add_node("profiler",       profiler)
builder.add_node("sql_writer",     sql_writer)
builder.add_node("executor",    executor)
builder.add_node("verifier",       verifier)

builder.add_edge(START, "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {"load_file_node": "load_file_node", "profiler": "profiler",
     "sql_writer": "sql_writer", END: END},
)

builder.add_edge("load_file_node", END)
builder.add_edge("profiler",       "sql_writer")
builder.add_edge("sql_writer",     "executor")

builder.add_conditional_edges(
    "executor",
    route_after_execute,
    {"verifier": "verifier", "orchestrator": "orchestrator"},
)

builder.add_conditional_edges(
    "verifier",
    route_after_verifier,
    {"orchestrator": "orchestrator", "sql_writer": "sql_writer"},
)

graph = builder.compile()
graph.name = "DuckDB Analytics Agent"
