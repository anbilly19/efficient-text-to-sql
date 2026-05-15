"""
db/catalog.py -- High-level helpers for the three metadata tables that power
multi-table support:  _column_catalog, _relationships, _table_context.

All write operations are idempotent (ON CONFLICT ... DO UPDATE) so they can be
called during re-ingestion without leaving stale data.
"""
from __future__ import annotations

import json
import re
from typing import Any

import duckdb

from agent.database import get_connection


# ---------------------------------------------------------------------------
# _table_context helpers
# ---------------------------------------------------------------------------

def upsert_table_context(
    dataset_name: str,
    summary: str,
    grain: str = "",
    tags: list[str] | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Insert or replace the _table_context row for *dataset_name*."""
    conn = conn or get_connection()
    tags_json = json.dumps(tags or [])
    conn.execute(
        """
        INSERT INTO _table_context (dataset_name, summary, grain, tags, updated_at)
        VALUES (?, ?, ?, ?, now())
        ON CONFLICT (dataset_name)
        DO UPDATE SET summary    = excluded.summary,
                      grain      = excluded.grain,
                      tags       = excluded.tags,
                      updated_at = now()
        """,
        [dataset_name, summary, grain, tags_json],
    )


def get_table_context(dataset_name: str, conn: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Return the _table_context row for *dataset_name* as a dict."""
    conn = conn or get_connection()
    row = conn.execute(
        "SELECT summary, grain, tags FROM _table_context WHERE dataset_name = ?",
        [dataset_name],
    ).fetchone()
    if not row:
        return {"summary": "", "grain": "", "tags": []}
    return {
        "summary": row[0] or "",
        "grain": row[1] or "",
        "tags": json.loads(row[2]) if row[2] else [],
    }


# ---------------------------------------------------------------------------
# _column_catalog helpers
# ---------------------------------------------------------------------------

def set_join_key(
    dataset_name: str,
    column_name: str,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> None:
    conn = conn or get_connection()
    conn.execute(
        "UPDATE _column_catalog SET is_join_key = TRUE "
        "WHERE dataset_name = ? AND column_name = ?",
        [dataset_name, column_name],
    )


def set_column_description(
    dataset_name: str,
    column_name: str,
    description: str,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> None:
    conn = conn or get_connection()
    conn.execute(
        "UPDATE _column_catalog SET description = ? "
        "WHERE dataset_name = ? AND column_name = ?",
        [description, dataset_name, column_name],
    )


def get_column_catalog(dataset_name: str, conn: duckdb.DuckDBPyConnection | None = None) -> list[dict]:
    conn = conn or get_connection()
    rows = conn.execute(
        """
        SELECT column_name, column_type, is_metric, is_dimension, is_join_key,
               description, sample_values
        FROM _column_catalog
        WHERE dataset_name = ?
        ORDER BY column_name
        """,
        [dataset_name],
    ).fetchall()
    return [
        {
            "column_name": r[0],
            "column_type": r[1],
            "is_metric": r[2],
            "is_dimension": r[3],
            "is_join_key": r[4],
            "description": r[5],
            "sample_values": json.loads(r[6]) if r[6] else [],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# _relationships helpers
# ---------------------------------------------------------------------------

def list_relationships(
    tables: list[str] | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[dict]:
    conn = conn or get_connection()
    if tables:
        rows = conn.execute(
            """
            SELECT left_table, left_column, right_table, right_column, cardinality, description
            FROM _relationships
            WHERE left_table = ANY(?) AND right_table = ANY(?)
            ORDER BY relationship_id
            """,
            [tables, tables],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT left_table, left_column, right_table, right_column, cardinality, description
            FROM _relationships
            ORDER BY relationship_id
            """
        ).fetchall()
    return [
        {
            "left_table":   r[0],
            "left_column":  r[1],
            "right_table":  r[2],
            "right_column": r[3],
            "cardinality":  r[4],
            "description":  r[5],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Alias resolution helper
# ---------------------------------------------------------------------------

def _extract_alias_map(sql: str) -> dict[str, str]:
    """Return {alias: table_name} for every FROM/JOIN clause in *sql*.

    Handles both quoted and unquoted table names and optional AS keyword:
        FROM sales_rep_targets t
        FROM sales_rep_targets AS t
        JOIN "sales1000" sa
        JOIN sales1000 AS sa
    CTE names (introduced by WITH <name> AS (...)) are intentionally excluded
    because they shadow actual table names and should not resolve to a
    registered table.
    """
    # Strip CTE definitions so their names don't pollute the alias map.
    # A CTE looks like:  <name> AS ( ... )
    # We remove the WITH block by scanning for balanced parentheses.
    sql_stripped = sql
    with_match = re.match(r'\bWITH\b', sql_stripped, re.IGNORECASE)
    if with_match:
        # Walk past all CTE definitions (balanced parens)
        i = with_match.end()
        while i < len(sql_stripped):
            if sql_stripped[i] == '(':
                depth = 1
                i += 1
                while i < len(sql_stripped) and depth:
                    if sql_stripped[i] == '(':  depth += 1
                    elif sql_stripped[i] == ')': depth -= 1
                    i += 1
                # skip optional comma between CTEs
                j = i
                while j < len(sql_stripped) and sql_stripped[j] in (' ', '\t', '\n', ','):
                    j += 1
                # if next non-space token looks like another CTE (word AS (),
                # keep looping; otherwise we've reached the main SELECT)
                peek = sql_stripped[j:j+50]
                if re.match(r'[\w"`]+\s+AS\s*\(', peek, re.IGNORECASE):
                    i = j
                    continue
                sql_stripped = sql_stripped[j:]
                break
            i += 1

    alias_map: dict[str, str] = {}
    # Match:  (FROM | JOIN)  [optional schema.]"table" | table  [AS]  alias
    pattern = re.compile(
        r'(?:FROM|JOIN)\s+'
        r'(?:[\w"`]+\.)?'        # optional schema prefix
        r'(["\`]?[\w]+["\`]?)'  # table name (group 1)
        r'(?:\s+AS)?\s+'
        r'([\w]+)'               # alias (group 2)
        r'(?=\s|$|\n|,|\))',
        re.IGNORECASE,
    )
    for m in pattern.finditer(sql_stripped):
        table = m.group(1).strip('"\`')
        alias = m.group(2).strip('"\`')
        # Exclude SQL keywords that can follow a table name
        if alias.upper() not in (
            'ON', 'WHERE', 'SET', 'INNER', 'LEFT', 'RIGHT',
            'FULL', 'CROSS', 'JOIN', 'GROUP', 'ORDER', 'HAVING',
            'LIMIT', 'UNION', 'EXCEPT', 'INTERSECT', 'SELECT',
        ):
            alias_map[alias] = table
    return alias_map


# ---------------------------------------------------------------------------
# _relationships validation
# ---------------------------------------------------------------------------

def validate_join_in_sql(sql: str, conn: duckdb.DuckDBPyConnection | None = None) -> list[str]:
    """Check every ON clause in *sql* against registered _relationships.

    Table aliases (e.g. ``s``, ``sa``) are resolved to their real table names
    before the lookup, so aliased JOINs are correctly validated.

    Returns a list of warning strings (empty == all joins look valid).
    """
    conn = conn or get_connection()
    warnings: list[str] = []

    # Build known-valid join pairs from _relationships (both directions).
    rels = conn.execute(
        "SELECT left_table, left_column, right_table, right_column FROM _relationships"
    ).fetchall()
    known: set[tuple[str, str, str, str]] = set()
    for lt, lc, rt, rc in rels:
        known.add((lt, lc, rt, rc))
        known.add((rt, rc, lt, lc))

    # Resolve aliases → real table names.
    alias_map = _extract_alias_map(sql)

    on_pattern = re.compile(
        r'\bON\b\s+([\w"`]+)\.([\w"`]+)\s*=\s*([\w"`]+)\.([\w"`]+)',
        re.IGNORECASE | re.MULTILINE,
    )
    for m in on_pattern.finditer(sql):
        raw_t1 = m.group(1).strip('`"')
        c1      = m.group(2).strip('`"')
        raw_t2 = m.group(3).strip('`"')
        c2      = m.group(4).strip('`"')

        # Resolve aliases; fall back to the raw token if no alias is found.
        t1 = alias_map.get(raw_t1, raw_t1)
        t2 = alias_map.get(raw_t2, raw_t2)

        if (t1, c1, t2, c2) not in known:
            warnings.append(
                f"JOIN key not in _relationships: "
                f"{t1}.{c1} = {t2}.{c2}"
                + (f" (via aliases {raw_t1}, {raw_t2})" if raw_t1 != t1 or raw_t2 != t2 else "")
                + ". Ensure the relationship is registered or correct the join columns."
            )
    return warnings
