"""LangGraph StateGraph assembly.

Routing (simplified):
  load       -> load_file_node -> END
  chitchat   -> END
  analytics  -> sql_writer -> execute_sql -> orchestrator (synthesis) -> END
               profiler -> sql_writer -> execute_sql -> verifier -> orchestrator -> END

Verifier is ONLY used when a profile step precedes the sql step.
For plain single-step SQL questions, execute_sql goes straight to orchestrator.
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


def _next_pending_step(state: AnalyticsState) -> PlanStep | None:
    return next((s for s in state.plan if s.status == "pending"), None)


def route_after_orchestrator(
    state: AnalyticsState,
) -> Literal["load_file_node", "profiler", "sql_writer", "__end__"]:
    if state.load_file_path:
        return "load_file_node"
    step = _next_pending_step(state)
    if step is not None:
        return "profiler" if step.type == "profile" else "sql_writer"
    return END


def route_after_execute(
    state: AnalyticsState,
) -> Literal["verifier", "orchestrator"]:
    """Skip verifier for simple single-sql plans; only verify when profiling preceded."""
    has_profile_step = any(s.type == "profile" for s in state.plan)
    next_step = _next_pending_step(state)
    # Use verifier only if there was profiling OR more steps remain
    if has_profile_step or next_step is not None:
        return "verifier"
    return "orchestrator"


def route_after_verifier(
    state: AnalyticsState,
) -> Literal["orchestrator", "sql_writer"]:
    if state.verification_verdict == "fail" and state.retry_count < 2:
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
builder.add_node("execute_sql",    execute_sql)
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
builder.add_edge("sql_writer",     "execute_sql")

builder.add_conditional_edges(
    "execute_sql",
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
