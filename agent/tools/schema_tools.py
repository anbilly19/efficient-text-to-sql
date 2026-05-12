"""
tools/schema_tools.py — LangChain tools for the orchestrator's table-selection
pre-filter and for the verifier's relationship check.

Design:
  select_tables       — keyword-based pre-filter (swappable with vector lookup)
  get_relationships   — return all registered join paths for a list of tables
  validate_joins      — check that a SQL string only uses registered join keys
"""
from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from agent.database import get_connection, get_table_summaries
from agent.db.catalog import list_relationships, validate_join_in_sql


@tool
def select_tables(
    query: Annotated[str, "The user's natural-language question"],
) -> str:
    """
    Return a JSON list of table names that are likely needed to answer *query*.

    Strategy (v1 — keyword pre-filter):
      1. Load all table names + their _table_context summaries/tags.
      2. Score each table by counting query-word hits in the summary + tag list.
      3. Return every table whose score > 0, or all tables when no hits.

    This interface is designed so that v2 can replace step 2 with a vector
    embedding lookup without changing the tool signature.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT dr.dataset_name, tc.summary, tc.tags "
        "FROM _data_registry dr "
        "LEFT JOIN _table_context tc ON tc.dataset_name = dr.dataset_name"
    ).fetchall()

    if not rows:
        return json.dumps([])

    q_words = set(query.lower().split())

    scored: list[tuple[int, str]] = []
    for dataset_name, summary, tags_json in rows:
        text = " ".join([
            dataset_name,
            summary or "",
            " ".join(json.loads(tags_json) if tags_json else []),
        ]).lower()
        score = sum(1 for w in q_words if w in text)
        scored.append((score, dataset_name))

    # Keep tables with a positive hit; fall back to all tables
    selected = [name for score, name in scored if score > 0]
    if not selected:
        selected = [name for _, name in scored]

    return json.dumps(selected)


@tool
def get_relationships(
    tables: Annotated[list[str], "Table names to retrieve join relationships for"],
) -> str:
    """
    Return a JSON list of all registered relationships involving *tables*.

    Each entry has keys: left_table, left_column, right_table, right_column,
    cardinality, description.
    """
    rels = list_relationships(tables=tables)
    return json.dumps(rels)


@tool
def validate_joins(
    sql: Annotated[str, "The SQL query to validate"],
) -> str:
    """
    Check that every JOIN in *sql* uses a column pair registered in _relationships.

    Returns a JSON object:
      {"ok": true}                         — all joins are valid
      {"ok": false, "warnings": ["...", ...]}  — one or more unregistered join keys
    """
    warnings = validate_join_in_sql(sql)
    if warnings:
        return json.dumps({"ok": False, "warnings": warnings})
    return json.dumps({"ok": True})
