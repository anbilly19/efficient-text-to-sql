"""LangGraph StateGraph assembly – this is what LangGraph Studio loads.

Exported symbol: `graph`  (referenced in langgraph.json)
"""
from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import execute_sql, orchestrator, profiler, sql_writer, verifier
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
) -> Literal["set_step", "__end__"]:
    """Go to set_step if work remains, else END (final answer already written)."""
    if state.final_answer:
        return END
    if _next_pending_step(state) is None:
        return END
    return "set_step"


def route_set_step(
    state: AnalyticsState,
) -> Literal["profiler", "sql_writer"]:
    """Dispatch to profiler or sql_writer based on the current step type."""
    step = state.current_step
    if step and step.type == "profile":
        return "profiler"
    return "sql_writer"


def route_after_verifier(
    state: AnalyticsState,
) -> Literal["orchestrator", "sql_writer"]:
    """Retry SQL on failure; otherwise return to orchestrator."""
    if state.verification_verdict == "fail" and state.last_sql:
        return "sql_writer"
    return "orchestrator"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

builder = StateGraph(AnalyticsState)

# Nodes
builder.add_node("orchestrator", orchestrator)
builder.add_node("set_step",     _set_current_step)
builder.add_node("profiler",     profiler)
builder.add_node("sql_writer",   sql_writer)
builder.add_node("execute_sql",  execute_sql)
builder.add_node("verifier",     verifier)

# Edges
builder.add_edge(START, "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {"set_step": "set_step", END: END},
)

builder.add_conditional_edges(
    "set_step",
    route_set_step,
    {"profiler": "profiler", "sql_writer": "sql_writer"},
)

builder.add_edge("profiler",    "orchestrator")   # profiler always returns to planner
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

graph = builder.compile(
    checkpointer=MemorySaver(),  # in-memory multi-turn state for Studio dev
)
graph.name = "DuckDB Analytics Agent"
