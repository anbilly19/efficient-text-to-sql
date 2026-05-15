"""
LangGraph node functions.

All nodes are generic — they read from metadata tables at runtime.
No domain-specific logic, no hardcoded column names, no file-type awareness.
"""

import json
import re
from typing import Any, TypedDict

import anthropic
from langgraph.graph import StateGraph, END

from agent.core.database import (
    get_all_tables, get_schema_context, get_semantic_map,
    get_query_rules, run_sql,
)
from agent.ingestion.loader import load_file
from agent.prompts.nodes import (
    ORCHESTRATOR_SYSTEM, SQL_WRITER_SYSTEM, VERIFIER_SYSTEM,
    PROFILER_SYSTEM, FINAL_ANSWER_SYSTEM,
    build_sql_writer_context, build_verifier_context,
)
from agent.tools.intent import classify, Intent, ClassifiedIntent
from agent.tools.resolver import resolve_aliases, semantic_search

_client = anthropic.Anthropic()
_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AnalyticsState(TypedDict):
    user_query: str
    intent: str
    active_table: str | None
    plan: list[dict]
    current_step: dict
    resolved_columns: dict
    last_sql: str
    last_result: list[dict]
    verification_verdict: str
    verification_feedback: str
    corrected_sql: str | None
    final_answer: str
    error: str
    messages: list[dict]


def _initial_state(query: str) -> AnalyticsState:
    return AnalyticsState(
        user_query=query,
        intent="",
        active_table=None,
        plan=[],
        current_step={},
        resolved_columns={},
        last_sql="",
        last_result=[],
        verification_verdict="",
        verification_feedback="",
        corrected_sql=None,
        final_answer="",
        error="",
        messages=[],
    )


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _llm(system: str, user: str, max_tokens: int = 2000) -> str:
    resp = _client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _parse_json(raw: str) -> dict | list:
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Meta-query response builders (structural, not domain-specific)
# ---------------------------------------------------------------------------

def _build_schema_reply(table_name: str) -> str:
    ctx = get_schema_context(table_name)
    cols = ctx["columns"]
    lines = [f"## Schema: {table_name}\n"]
    for role in ["dimension", "period", "metric", "yoy_delta", "prior_period", "unknown"]:
        group = [c for c in cols if c["column_role"] == role]
        if group:
            lines.append(f"**{role.upper()} columns**")
            for c in group:
                lines.append(
                    f"  - `{c['column_name']}` ({c['dtype']}, "
                    f"null={c['null_rate']:.1%}, unique={c['n_unique']:,})"
                )
    return "\n".join(lines)


def _build_meta_reply() -> str:
    tables = get_all_tables()
    if not tables:
        return "No tables are currently loaded. Use `load <path> as <name>` to load a file."
    lines = ["## Loaded tables\n"]
    for t in tables:
        reg = run_sql("SELECT row_count, col_count, source_file FROM _data_registry WHERE table_name = ?", [t])
        ctx = run_sql("SELECT grain, cy_label, summary FROM _table_context WHERE table_name = ?", [t])
        r = reg[0] if reg else {}
        c = ctx[0] if ctx else {}
        lines.append(
            f"**{t}** — {r.get('row_count', '?'):,} rows, {r.get('col_count', '?')} cols, "
            f"grain={c.get('grain', '?')}, CY={c.get('cy_label', '?')}\n"
            f"  Source: {r.get('source_file', '?')}\n"
            f"  {c.get('summary', '')}"
        )
    return "\n\n".join(lines)


def _build_distinct_reply(table_name: str, column: str) -> str:
    rows = run_sql(f'SELECT DISTINCT "{column}" FROM "{table_name}" ORDER BY 1 LIMIT 50')
    values = [list(r.values())[0] for r in rows]
    return f"Distinct values in `{column}` ({table_name}):\n" + "\n".join(f"  - {v}" for v in values)


def _build_grain_reply(table_name: str) -> str:
    ctx = run_sql(
        "SELECT grain, period_column, cy_label, py_label FROM _table_context WHERE table_name = ?",
        [table_name],
    )
    if not ctx:
        return f"No context found for table `{table_name}`."
    c = ctx[0]
    return (
        f"**{table_name}** grain: `{c['grain']}`\n"
        f"Period column: `{c['period_column']}`\n"
        f"Current year: `{c['cy_label']}`\n"
        f"Prior year: `{c['py_label']}`"
    )


def _build_semantic_lookup_reply(term: str, table_name: str | None) -> str:
    results = semantic_search(term, table_name, top_k=8)
    if not results:
        return f"No semantic map entries found for `{term}`."
    lines = [f"Semantic map results for `{term}`:\n"]
    for r in results:
        lines.append(
            f"  - `{r['alias']}` → **{r['canonical_col']}** "
            f"[{r['table_name']}] ({r['alias_lang']})\n"
            f"    {r['description']}"
        )
    return "\n".join(lines)


def _build_empty_sql_reply(concept: str, table_name: str) -> str:
    dims = run_sql(
        "SELECT column_name FROM _column_catalog WHERE table_name = ? AND column_role = 'dimension'",
        [table_name],
    )
    metrics = run_sql(
        "SELECT column_name FROM _column_catalog WHERE table_name = ? AND column_role = 'metric' LIMIT 5",
        [table_name],
    )
    dim_list = [r["column_name"] for r in dims]
    met_list = [r["column_name"] for r in metrics]
    return (
        f"No column matching `{concept}` was found in `{table_name}`.\n\n"
        f"**Dimension columns**: {', '.join(dim_list)}\n"
        f"**Sample metric columns**: {', '.join(met_list)}\n\n"
        "Please rephrase your question using the column names above."
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def orchestrator_node(state: AnalyticsState) -> AnalyticsState:
    """Route and plan — reads metadata, no domain hardcoding."""
    query = state["user_query"]
    classified = classify(query)
    table = classified.table_name

    # --- Structural fast-paths (format-agnostic) ---

    if classified.intent == Intent.LOAD_FILE:
        # Parse: "load <path> as <name>"
        m = re.match(r"load\s+(\S+)\s+as\s+(\w+)", query.strip(), re.IGNORECASE)
        if m:
            path, name = m.group(1), m.group(2)
            try:
                result = load_file(path, name)
                state["final_answer"] = (
                    f"Loaded `{name}` from `{path}`: "
                    f"{result['rows']:,} rows, {result['columns']} columns. "
                    f"Grain: {result['grain']}. CY: {result['cy_label']}."
                )
            except Exception as e:
                state["error"] = str(e)
                state["final_answer"] = f"Error loading file: {e}"
        return state

    if classified.intent == Intent.META_QUERY:
        state["final_answer"] = _build_meta_reply()
        return state

    if classified.intent == Intent.SCHEMA_LOOKUP:
        state["final_answer"] = _build_schema_reply(table) if table else _build_meta_reply()
        return state

    if classified.intent == Intent.DISTINCT_VALUES:
        col = classified.extra.get("column")
        if table and col:
            state["final_answer"] = _build_distinct_reply(table, col)
        return state

    if classified.intent == Intent.GRAIN_QUERY:
        state["final_answer"] = _build_grain_reply(table) if table else "Please specify a table."
        return state

    if classified.intent == Intent.SEMANTIC_LOOKUP:
        term = classified.extra.get("term", query)
        state["final_answer"] = _build_semantic_lookup_reply(term, table)
        return state

    if classified.intent == Intent.ADVERSARIAL:
        state["final_answer"] = _build_empty_sql_reply(query, table or "")
        return state

    # --- Analytics path: build a plan ---
    state["active_table"] = table
    state["intent"] = classified.intent.value

    # Resolve aliases against semantic map
    if table:
        _, resolved = resolve_aliases(query, table)
        state["resolved_columns"] = resolved

    # Simple single-step plan
    state["plan"] = [
        {"id": 1, "type": "sql_writer", "description": "Generate SQL", "status": "pending"},
        {"id": 2, "type": "verify",     "description": "Verify result", "status": "pending"},
    ]
    state["current_step"] = state["plan"][0]
    return state


def sql_writer_node(state: AnalyticsState) -> AnalyticsState:
    table = state["active_table"]
    if not table:
        state["final_answer"] = "No table selected."
        return state

    schema_ctx = get_schema_context(table)
    query_rules = get_query_rules(table)
    table_ctx_rows = run_sql("SELECT * FROM _table_context WHERE table_name = ?", [table])
    table_ctx = table_ctx_rows[0] if table_ctx_rows else {}

    user_msg = build_sql_writer_context(
        question=state["user_query"],
        table_name=table,
        schema_context=schema_ctx,
        resolved_columns=state["resolved_columns"],
        query_rules=query_rules,
        table_ctx=table_ctx,
    )
    raw = _llm(SQL_WRITER_SYSTEM, user_msg)
    try:
        parsed = _parse_json(raw)
    except json.JSONDecodeError:
        state["error"] = f"SQL Writer returned non-JSON: {raw[:200]}"
        return state

    sql = parsed.get("sql", "")
    if not sql:
        concept = state["user_query"]
        state["final_answer"] = _build_empty_sql_reply(concept, table)
        return state

    state["last_sql"] = sql
    # Mark step done
    for step in state["plan"]:
        if step["type"] == "sql_writer":
            step["status"] = "done"
    return state


def execute_sql_node(state: AnalyticsState) -> AnalyticsState:
    sql = state.get("corrected_sql") or state.get("last_sql", "")
    if not sql:
        return state
    try:
        rows = run_sql(sql)
        state["last_result"] = rows
        state["last_sql"] = sql
        state["corrected_sql"] = None
    except Exception as e:
        state["error"] = f"SQL execution error: {e}"
        state["last_result"] = []
    return state


def verifier_node(state: AnalyticsState) -> AnalyticsState:
    table = state["active_table"]
    query_rules = get_query_rules(table) if table else []
    table_ctx_rows = run_sql("SELECT * FROM _table_context WHERE table_name = ?", [table]) if table else []
    table_ctx = table_ctx_rows[0] if table_ctx_rows else {}

    user_msg = build_verifier_context(
        question=state["user_query"],
        sql=state["last_sql"],
        result_rows=state["last_result"],
        query_rules=query_rules,
        table_ctx=table_ctx,
    )
    raw = _llm(VERIFIER_SYSTEM, user_msg)
    try:
        parsed = _parse_json(raw)
    except json.JSONDecodeError:
        state["verification_verdict"] = "warning"
        state["verification_feedback"] = "Could not parse verifier response."
        return state

    state["verification_verdict"] = parsed.get("verdict", "pass")
    state["verification_feedback"] = parsed.get("feedback", "")
    state["corrected_sql"] = parsed.get("corrected_sql")

    for step in state["plan"]:
        if step["type"] == "verify":
            step["status"] = "done"
    return state


def final_answer_node(state: AnalyticsState) -> AnalyticsState:
    if state["final_answer"]:
        return state

    result_block = json.dumps(state["last_result"][:20], ensure_ascii=False)
    user_msg = (
        f"Question: {state['user_query']}\n\n"
        f"SQL used: {state['last_sql']}\n\n"
        f"Result: {result_block}\n\n"
        f"Verification: {state['verification_feedback']}"
    )
    state["final_answer"] = _llm(FINAL_ANSWER_SYSTEM, user_msg, max_tokens=400)
    return state


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_after_orchestrator(state: AnalyticsState) -> str:
    if state["final_answer"]:
        return "done"
    if state["plan"]:
        return "sql_writer"
    return "done"


def route_after_verify(state: AnalyticsState) -> str:
    verdict = state.get("verification_verdict", "pass")
    if verdict == "fail" and state.get("corrected_sql"):
        return "execute_corrected"
    return "final_answer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    graph = StateGraph(AnalyticsState)

    graph.add_node("orchestrator",   orchestrator_node)
    graph.add_node("sql_writer",     sql_writer_node)
    graph.add_node("execute_sql",    execute_sql_node)
    graph.add_node("verifier",       verifier_node)
    graph.add_node("final_answer",   final_answer_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges("orchestrator", route_after_orchestrator, {
        "sql_writer": "sql_writer",
        "done":       END,
    })
    graph.add_edge("sql_writer",   "execute_sql")
    graph.add_edge("execute_sql",  "verifier")
    graph.add_conditional_edges("verifier", route_after_verify, {
        "execute_corrected": "execute_sql",
        "final_answer":      "final_answer",
    })
    graph.add_edge("final_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public run function
# ---------------------------------------------------------------------------

def run_query(query: str) -> str:
    graph = build_graph()
    state = _initial_state(query)
    final_state = graph.invoke(state)
    return final_state.get("final_answer") or final_state.get("error") or "No answer."
