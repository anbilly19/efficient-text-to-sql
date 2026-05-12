"""
db/catalog.py — High-level helpers for the three metadata tables that power
multi-table support:  _column_catalog, _relationships, _table_context.

All write operations are idempotent (ON CONFLICT … DO UPDATE) so they can be
called during re-ingestion without leaving stale data.
"""
from __future__ import annotations

import json
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
        VALUES (?, ?, ?, ?, current_timestamp)
        ON CONFLICT (dataset_name)
        DO UPDATE SET summary    = excluded.summary,
                      grain      = excluded.grain,
                      tags       = excluded.tags,
                      updated_at = current_timestamp
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
    """Mark *column_name* in *dataset_name* as a join key."""
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
    """Set a human-readable description for a catalog column."""
    conn = conn or get_connection()
    conn.execute(
        "UPDATE _column_catalog SET description = ? "
        "WHERE dataset_name = ? AND column_name = ?",
        [description, dataset_name, column_name],
    )


def get_column_catalog(dataset_name: str, conn: duckdb.DuckDBPyConnection | None = None) -> list[dict]:
    """Return the full _column_catalog for *dataset_name* as a list of dicts."""
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
    """Return all relationships, optionally filtered to a list of table names."""
    conn = conn or get_connection()
    if tables:
        rows = conn.execute(
            """
            SELECT left_table, left_column, right_table, right_column, cardinality, description
            FROM _relationships
            WHERE left_table = ANY(?) OR right_table = ANY(?)
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
            "left_table": r[0],
            "left_column": r[1],
            "right_table": r[2],
            "right_column": r[3],
            "cardinality": r[4],
            "description": r[5],
        }
        for r in rows
    ]


def validate_join_in_sql(sql: str, conn: duckdb.DuckDBPyConnection | None = None) -> list[str]:
    """Best-effort check: return a list of warning strings for JOIN patterns
    whose key pairs are NOT registered in _relationships.

    Only catches simple  t1.col = t2.col  patterns in ON clauses.
    Returns [] when the SQL is clean (or the heuristic cannot parse it).
    """
    import re
    conn = conn or get_connection()
    warnings: list[str] = []

    # Collect all known join pairs (both directions)
    rels = conn.execute(
        "SELECT left_table, left_column, right_table, right_column FROM _relationships"
    ).fetchall()
    known: set[tuple[str, str, str, str]] = set()
    for lt, lc, rt, rc in rels:
        known.add((lt, lc, rt, rc))
        known.add((rt, rc, lt, lc))  # bidirectional

    # Find ON clause patterns like:  tbl1.col1 = tbl2.col2
    on_pattern = re.compile(
        r'(?:^|\s)ON\s+([\w"`]+)\.([\w"`]+)\s*=\s*([\w"`]+)\.([\w"`]+)',
        re.IGNORECASE | re.MULTILINE,
    )
    for m in on_pattern.finditer(sql):
        t1 = m.group(1).strip('`"')
        c1 = m.group(2).strip('`"')
        t2 = m.group(3).strip('`"')
        c2 = m.group(4).strip('`"')
        if (t1, c1, t2, c2) not in known:
            warnings.append(
                f"JOIN key not in _relationships: {t1}.{c1} = {t2}.{c2}. "
                "Ensure the relationship is registered or use an explicit JOIN."
            )
    return warnings
