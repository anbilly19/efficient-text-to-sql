"""LangGraph StateGraph assembly.

Intent routing:
  load      → load_file_node → END
  analytics → profiler | sql_writer → execute_sql → verifier → orchestrator (synthesis) → END
  chitchat  → END
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


def route_after_orchestrator(
    state: AnalyticsState,
) -> Literal["load_file_node", "profiler", "sql_writer", "__end__"]:
    # File load
    if state.load_file_path:
        return "load_file_node"
    # Analytics: route directly to the right agent node
    step = _next_pending_step(state)
    if step is not None:
        return "profiler" if step.type == "profile" else "sql_writer"
    # Chitchat / synthesis done
    if state.final_answer:
        return END
    return END


def route_after_verifier(
    state: AnalyticsState,
) -> Literal["orchestrator", "sql_writer"]:
    if state.verification_verdict == "fail" and state.last_sql:
        return "sql_writer"
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
builder.add_edge("profiler",       "orchestrator")  # profiler → re-route (may go to sql_writer next)
builder.add_edge("sql_writer",     "execute_sql")
builder.add_edge("execute_sql",    "verifier")

builder.add_conditional_edges(
    "verifier",
    route_after_verifier,
    {"orchestrator": "orchestrator", "sql_writer": "sql_writer"},
)

graph = builder.compile()
graph.name = "DuckDB Analytics Agent"
