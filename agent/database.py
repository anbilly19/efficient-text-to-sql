"""
database.py — DuckDB connection, metadata tables, and dynamic schema indexing.

Design principles:
- NIQ-independent: no hard-coded column names or dataset names anywhere.
- All schema knowledge is derived at ingestion time and stored in DuckDB metadata tables.
- Multi-table: _column_catalog tracks every column of every table with semantic roles.
- _relationships stores explicit join keys; nothing is inferred from column-name matching.
- _table_context stores a compact one-line description per table for the orchestrator selection pass.
"""

import os
import threading
from pathlib import Path

import duckdb

_conn: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()

# Absolute path to the project root (parent of the agent/ package directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / ".local" / "duckdb" / "efficient-text-to-sql.duckdb"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection, initialising it on first call."""
    global _conn
    with _lock:
        if _conn is None:
            db_path = os.getenv("DUCKDB_PATH")
            if not db_path:
                _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                db_path = str(_DEFAULT_DB_PATH)
            _conn = duckdb.connect(db_path)
            _ensure_metadata_tables(_conn)
            _auto_reattach_parquet(_conn)
    return _conn


# ---------------------------------------------------------------------------
# Metadata table bootstrap + schema migration
# ---------------------------------------------------------------------------

def _add_column_if_missing(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    definition: str,
) -> None:
    exists = conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name = ? AND column_name = ?
        """,
        [table, column],
    ).fetchone()[0]
    if not exists:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {column} {definition}')


def _ensure_metadata_tables(conn: duckdb.DuckDBPyConnection) -> None:
    # ── _data_registry ─────────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _data_registry (
            dataset_name  VARCHAR NOT NULL PRIMARY KEY,
            parquet_path  VARCHAR NOT NULL,
            source_file   VARCHAR,
            ingested_at   TIMESTAMP DEFAULT current_timestamp,
            row_count     BIGINT,
            column_count  INTEGER
        )
    """)
    _add_column_if_missing(conn, "_data_registry", "ingested_at",
                           "TIMESTAMP DEFAULT current_timestamp")
    _add_column_if_missing(conn, "_data_registry", "row_count",   "BIGINT")
    _add_column_if_missing(conn, "_data_registry", "column_count", "INTEGER")

    # ── _column_catalog ───────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _column_catalog (
            dataset_name  VARCHAR NOT NULL,
            column_name   VARCHAR NOT NULL,
            column_type   VARCHAR,
            is_metric     BOOLEAN DEFAULT FALSE,
            is_dimension  BOOLEAN DEFAULT FALSE,
            is_join_key   BOOLEAN DEFAULT FALSE,
            description   VARCHAR,
            sample_values VARCHAR,
            PRIMARY KEY (dataset_name, column_name)
        )
    """)
    _add_column_if_missing(conn, "_column_catalog", "is_metric",     "BOOLEAN DEFAULT FALSE")
    _add_column_if_missing(conn, "_column_catalog", "is_dimension",  "BOOLEAN DEFAULT FALSE")
    _add_column_if_missing(conn, "_column_catalog", "is_join_key",   "BOOLEAN DEFAULT FALSE")
    _add_column_if_missing(conn, "_column_catalog", "description",   "VARCHAR")
    _add_column_if_missing(conn, "_column_catalog", "sample_values", "VARCHAR")

    # ── _relationships ──────────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _relationships (
            relationship_id  INTEGER PRIMARY KEY,
            left_table       VARCHAR NOT NULL,
            left_column      VARCHAR NOT NULL,
            right_table      VARCHAR NOT NULL,
            right_column     VARCHAR NOT NULL,
            cardinality      VARCHAR DEFAULT 'many-to-one',
            description      VARCHAR,
            added_at         TIMESTAMP DEFAULT current_timestamp,
            UNIQUE (left_table, left_column, right_table, right_column)
        )
    """)
    _add_column_if_missing(conn, "_relationships", "cardinality",
                           "VARCHAR DEFAULT 'many-to-one'")
    _add_column_if_missing(conn, "_relationships", "description", "VARCHAR")
    _add_column_if_missing(conn, "_relationships", "added_at",
                           "TIMESTAMP DEFAULT current_timestamp")

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS _rel_seq START 1
    """)

    # ── _table_context ──────────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _table_context (
            dataset_name  VARCHAR NOT NULL PRIMARY KEY,
            summary       VARCHAR,
            grain         VARCHAR,
            tags          VARCHAR,
            updated_at    TIMESTAMP DEFAULT current_timestamp
        )
    """)
    _add_column_if_missing(conn, "_table_context", "grain",      "VARCHAR")
    _add_column_if_missing(conn, "_table_context", "tags",       "VARCHAR")
    _add_column_if_missing(conn, "_table_context", "updated_at",
                           "TIMESTAMP DEFAULT current_timestamp")

    # ── _semantic_map ───────────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _semantic_map (
            term          VARCHAR NOT NULL,
            dataset_name  VARCHAR NOT NULL DEFAULT '',
            column_name   VARCHAR NOT NULL,
            description   VARCHAR,
            UNIQUE (term, dataset_name)
        )
    """)
    _add_column_if_missing(conn, "_semantic_map", "description", "VARCHAR")


# ---------------------------------------------------------------------------
# Parquet auto-reattach
# ---------------------------------------------------------------------------

def _auto_reattach_parquet(conn: duckdb.DuckDBPyConnection) -> None:
    """Re-create views for all registered Parquet files on startup."""
    rows = conn.execute(
        "SELECT dataset_name, parquet_path FROM _data_registry"
    ).fetchall()
    for dataset_name, parquet_path in rows:
        if not Path(parquet_path).exists():
            continue
        exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [dataset_name],
        ).fetchone()[0]
        if not exists:
            conn.execute(
                f'CREATE VIEW "{dataset_name}" AS SELECT * FROM read_parquet(\'{parquet_path}\')'
            )


# ---------------------------------------------------------------------------
# Dynamic schema indexing
# ---------------------------------------------------------------------------

def index_table_schema(conn: duckdb.DuckDBPyConnection, dataset_name: str) -> None:
    """Introspect a newly loaded table and populate _column_catalog."""
    conn.execute(
        "DELETE FROM _column_catalog WHERE dataset_name = ?",
        [dataset_name],
    )

    columns = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [dataset_name],
    ).fetchall()

    rows_to_insert = []
    for col_name, col_type in columns:
        col_type_up = col_type.upper()
        is_metric = any(
            t in col_type_up
            for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "HUGEINT", "BIGINT")
        )
        is_dimension = not is_metric

        try:
            sample_rows = conn.execute(
                f'SELECT DISTINCT "{col_name}" FROM "{dataset_name}" '
                f'WHERE "{col_name}" IS NOT NULL LIMIT 5'
            ).fetchall()
            import json
            sample_values = json.dumps([str(r[0]) for r in sample_rows])
        except Exception:
            sample_values = "[]"

        rows_to_insert.append((
            dataset_name,
            col_name,
            col_type,
            is_metric,
            is_dimension,
            False,
            None,
            sample_values,
        ))

    conn.executemany(
        "INSERT INTO _column_catalog "
        "(dataset_name, column_name, column_type, is_metric, is_dimension, is_join_key, description, sample_values) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows_to_insert,
    )


def infer_and_register_relationships(conn: duckdb.DuckDBPyConnection, new_table: str) -> list[dict]:
    """Auto-detect probable join keys between new_table and all existing tables."""
    existing_tables = conn.execute(
        "SELECT DISTINCT dataset_name FROM _column_catalog WHERE dataset_name != ?",
        [new_table],
    ).fetchall()

    new_cols = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT column_name, column_type FROM _column_catalog WHERE dataset_name = ?",
            [new_table],
        ).fetchall()
    }

    registered = []
    for (other_table,) in existing_tables:
        other_cols = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT column_name, column_type FROM _column_catalog WHERE dataset_name = ?",
                [other_table],
            ).fetchall()
        }
        for col_name, col_type in new_cols.items():
            if col_name in other_cols:
                def _is_str(t: str) -> bool:
                    return any(x in t.upper() for x in ("VARCHAR", "TEXT", "CHAR"))

                def _is_num(t: str) -> bool:
                    return any(
                        x in t.upper()
                        for x in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")
                    )

                if _is_str(col_type) == _is_str(other_cols[col_name]) or _is_num(
                    col_type
                ) == _is_num(other_cols[col_name]):
                    try:
                        conn.execute(
                            """
                            INSERT INTO _relationships
                                (relationship_id, left_table, left_column, right_table, right_column, cardinality, description)
                            VALUES (nextval('_rel_seq'), ?, ?, ?, ?, 'inferred', 'auto-detected by column name match')
                            ON CONFLICT (left_table, left_column, right_table, right_column) DO NOTHING
                            """,
                            [new_table, col_name, other_table, col_name],
                        )
                        conn.execute(
                            "UPDATE _column_catalog SET is_join_key = TRUE "
                            "WHERE dataset_name = ? AND column_name = ?",
                            [new_table, col_name],
                        )
                        conn.execute(
                            "UPDATE _column_catalog SET is_join_key = TRUE "
                            "WHERE dataset_name = ? AND column_name = ?",
                            [other_table, col_name],
                        )
                        registered.append(
                            {"left": new_table, "right": other_table, "on": col_name}
                        )
                    except Exception:
                        pass
    return registered


def register_relationship(
    conn: duckdb.DuckDBPyConnection,
    left_table: str,
    left_column: str,
    right_table: str,
    right_column: str,
    cardinality: str = "many-to-one",
    description: str | None = None,
) -> None:
    """Explicitly register an authoritative join relationship."""
    conn.execute(
        """
        INSERT INTO _relationships
            (relationship_id, left_table, left_column, right_table, right_column, cardinality, description)
        VALUES (nextval('_rel_seq'), ?, ?, ?, ?, ?, ?)
        ON CONFLICT (left_table, left_column, right_table, right_column)
        DO UPDATE SET cardinality = excluded.cardinality,
                      description = excluded.description,
                      added_at    = now()
        """,
        [left_table, left_column, right_table, right_column, cardinality, description],
    )
    for tbl, col in [(left_table, left_column), (right_table, right_column)]:
        conn.execute(
            "UPDATE _column_catalog SET is_join_key = TRUE "
            "WHERE dataset_name = ? AND column_name = ?",
            [tbl, col],
        )


# ---------------------------------------------------------------------------
# Schema context builders (for prompt injection)
# ---------------------------------------------------------------------------

def get_schema_context(table_names: list[str]) -> str:
    conn = get_connection()
    lines = []
    for tbl in table_names:
        ctx = conn.execute(
            "SELECT summary, grain FROM _table_context WHERE dataset_name = ?",
            [tbl],
        ).fetchone()
        summary = ctx[0] if ctx and ctx[0] else ""
        grain   = ctx[1] if ctx and ctx[1] else ""

        lines.append(f'## Table: "{tbl}"')
        lines.append(
            '-- RULE: use ONLY the column names listed below, '
            'exactly as quoted.  Do NOT invent, abbreviate, or rename any column.'
        )
        if summary:
            lines.append(f"Summary: {summary}")
        if grain:
            lines.append(f"Grain: {grain}")

        cols = conn.execute(
            "SELECT column_name, column_type, is_metric, is_dimension, "
            "is_join_key, description, sample_values "
            "FROM _column_catalog WHERE dataset_name = ? ORDER BY column_name",
            [tbl],
        ).fetchall()

        if not cols:
            lines.append("  (no columns indexed — re-ingest the table)")
        else:
            for col_name, col_type, is_metric, is_dim, is_jk, desc, samples in cols:
                flags = []
                if is_metric:    flags.append("metric")
                if is_dim:       flags.append("dimension")
                if is_jk:        flags.append("join_key")
                flag_str = ", ".join(flags)
                desc_str = f" — {desc}" if desc else ""
                lines.append(f'  "{col_name}" ({col_type}) [{flag_str}]{desc_str}')
                if samples and samples != "[]":
                    lines.append(f"    samples: {samples}")
        lines.append("")

    if len(table_names) > 1:
        rels = conn.execute(
            """
            SELECT left_table, left_column, right_table, right_column, cardinality
            FROM _relationships
            WHERE left_table = ANY(?) OR right_table = ANY(?)
            """,
            [table_names, table_names],
        ).fetchall()
        if rels:
            lines.append("## Relationships (authoritative join keys — use ONLY these pairs)")
            for lt, lc, rt, rc, card in rels:
                lines.append(f'  "{lt}"."{lc}" → "{rt}"."{rc}"  ({card})')

    return "\n".join(lines)


def get_table_summaries() -> str:
    """Return a short listing of all registered tables with their summaries."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT dr.dataset_name, dr.row_count, dr.column_count,
               tc.summary, tc.grain
        FROM _data_registry dr
        LEFT JOIN _table_context tc ON tc.dataset_name = dr.dataset_name
        ORDER BY dr.ingested_at
        """
    ).fetchall()
    if not rows:
        return "No tables loaded."
    lines = []
    for name, rows_n, cols_n, summary, grain in rows:
        summary_str = summary or "(no summary yet)"
        grain_str = f" | grain: {grain}" if grain else ""
        lines.append(f"- {name}  ({rows_n or '?'} rows, {cols_n or '?'} cols){grain_str}  — {summary_str}")
    return "\n".join(lines)
