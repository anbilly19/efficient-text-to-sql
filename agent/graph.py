"""LangGraph StateGraph assembly.

Routing:
  intent_classifier (START)
    ingest_file   -> kg_ingest_node    -> END
    kg_group      -> kg_group_node     -> END
    kg_derived    -> kg_derived_node   -> END
    kg_hierarchy  -> kg_hierarchy_node -> END
    kg_review     -> kg_review_node    -> END
    analytics     -> orchestrator      -> ...

  orchestrator
    chitchat/done         -> END
    simple sql            -> sql_writer -> executor -> verifier -> orchestrator
    with profile step     -> profiler   -> sql_writer -> ...
    on fail               -> verifier   -> sql_writer (max 2 retries)
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
from agent.nodes_kg import (
    kg_ingest_node,
    kg_group_node,
    kg_derived_node,
    kg_hierarchy_node,
    kg_review_node,
)
from agent.intent_classifier import intent_classifier
from agent.state import AnalyticsState, PlanStep

MAX_RETRIES = 2
MAX_PLAN_STEPS = 5


def _next_pending_step(state: AnalyticsState) -> PlanStep | None:
    return next((s for s in state.plan if s.status == 'pending'), None)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_classifier(
    state: AnalyticsState,
) -> Literal[
    'kg_ingest_node', 'kg_group_node', 'kg_derived_node',
    'kg_hierarchy_node', 'kg_review_node', 'orchestrator',
]:
    intent = state.intent
    return {
        'ingest_file':  'kg_ingest_node',
        'kg_group':     'kg_group_node',
        'kg_derived':   'kg_derived_node',
        'kg_hierarchy': 'kg_hierarchy_node',
        'kg_review':    'kg_review_node',
    }.get(intent, 'orchestrator')


def route_after_orchestrator(
    state: AnalyticsState,
) -> Literal['load_file_node', 'profiler', 'sql_writer', '__end__']:
    if state.load_file_path:
        return 'load_file_node'
    if len(state.plan) > MAX_PLAN_STEPS:
        return END
    step = _next_pending_step(state)
    if step is not None:
        return 'profiler' if step.type == 'profile' else 'sql_writer'
    return END


def route_after_execute(
    state: AnalyticsState,
) -> Literal['verifier', 'orchestrator']:
    if state.error and not state.last_query_result:
        return 'orchestrator'
    return 'verifier'


def route_after_verifier(
    state: AnalyticsState,
) -> Literal['orchestrator', 'sql_writer']:
    if state.retry_count >= MAX_RETRIES:
        return 'orchestrator'
    if state.verification_verdict == 'fail':
        return 'sql_writer'
    if _next_pending_step(state) is not None:
        return 'sql_writer'
    return 'orchestrator'


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(AnalyticsState)

# Nodes
builder.add_node('intent_classifier',  intent_classifier)
builder.add_node('orchestrator',       orchestrator)
builder.add_node('kg_ingest_node',     kg_ingest_node)
builder.add_node('kg_group_node',      kg_group_node)
builder.add_node('kg_derived_node',    kg_derived_node)
builder.add_node('kg_hierarchy_node',  kg_hierarchy_node)
builder.add_node('kg_review_node',     kg_review_node)
builder.add_node('load_file_node',     load_file_node)
builder.add_node('profiler',           profiler)
builder.add_node('sql_writer',         sql_writer)
builder.add_node('executor',           executor)
builder.add_node('verifier',           verifier)

# Edges
builder.add_edge(START, 'intent_classifier')

builder.add_conditional_edges(
    'intent_classifier',
    route_after_classifier,
    {
        'kg_ingest_node':    'kg_ingest_node',
        'kg_group_node':     'kg_group_node',
        'kg_derived_node':   'kg_derived_node',
        'kg_hierarchy_node': 'kg_hierarchy_node',
        'kg_review_node':    'kg_review_node',
        'orchestrator':      'orchestrator',
    },
)

# KG terminal nodes
for _kg_node in ('kg_ingest_node', 'kg_group_node', 'kg_derived_node',
                  'kg_hierarchy_node', 'kg_review_node'):
    builder.add_edge(_kg_node, END)

# Analytics path
builder.add_edge('load_file_node', END)
builder.add_edge('profiler',       'sql_writer')
builder.add_edge('sql_writer',     'executor')

builder.add_conditional_edges(
    'orchestrator',
    route_after_orchestrator,
    {
        'load_file_node': 'load_file_node',
        'profiler':       'profiler',
        'sql_writer':     'sql_writer',
        END:              END,
    },
)

builder.add_conditional_edges(
    'executor',
    route_after_execute,
    {'verifier': 'verifier', 'orchestrator': 'orchestrator'},
)

builder.add_conditional_edges(
    'verifier',
    route_after_verifier,
    {'orchestrator': 'orchestrator', 'sql_writer': 'sql_writer'},
)

graph = builder.compile()
graph.name = 'DuckDB Analytics Agent'
