"""LangGraph StateGraph assembly – this is what LangGraph Studio loads.

Exported symbol: `graph`  (referenced in langgraph.json)

Intent routing from orchestrator:
  load      → load_file_node → END
  analytics → set_step → profiler | sql_writer → execute_sql → verifier → orchestrator
  chitchat  → END  (final_answer already set)
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


def _set_current_step(state: AnalyticsState) -> dict:
    """Pin the next pending step as current_step before routing to an agent."""
    step = _next_pending_step(state)
    return {"current_step": step}


def route_after_orchestrator(
    state: AnalyticsState,
) -> Literal["load_file_node", "set_step", "__end__"]:
    """Three-way routing based on orchestrator intent."""
    # File load intent – path was set, no plan built
    if state.load_file_path:
        return "load_file_node"
    # Final answer already written (chitchat or synthesis complete)
    if state.final_answer:
        return END
    # Analytics intent – plan was built
    if _next_pending_step(state) is not None:
        return "set_step"
    return END


def route_set_step(
    state: AnalyticsState,
) -> Literal["profiler", "sql_writer"]:
    step = state.current_step
    if step and step.type == "profile":
        return "profiler"
    return "sql_writer"


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

# Nodes
builder.add_node("orchestrator",   orchestrator)
builder.add_node("load_file_node", load_file_node)
builder.add_node("set_step",       _set_current_step)
builder.add_node("profiler",       profiler)
builder.add_node("sql_writer",     sql_writer)
builder.add_node("execute_sql",    execute_sql)
builder.add_node("verifier",       verifier)

# Edges
builder.add_edge(START, "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {"load_file_node": "load_file_node", "set_step": "set_step", END: END},
)

builder.add_edge("load_file_node", END)   # load always terminates the turn

builder.add_conditional_edges(
    "set_step",
    route_set_step,
    {"profiler": "profiler", "sql_writer": "sql_writer"},
)

builder.add_edge("profiler",    "orchestrator")
builder.add_edge("sql_writer",  "execute_sql")
builder.add_edge("execute_sql", "verifier")

builder.add_conditional_edges(
    "verifier",
    route_after_verifier,
    {"orchestrator": "orchestrator", "sql_writer": "sql_writer"},
)

# ---------------------------------------------------------------------------
# Compiled graph – exported for LangGraph Studio
# ---------------------------------------------------------------------------

graph = builder.compile()
graph.name = "DuckDB Analytics Agent"
