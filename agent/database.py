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
                os.makedirs("./.local/duckdb", exist_ok=True)
                db_path = "./.local/duckdb/efficient-text-to-sql.duckdb"
            _conn = duckdb.connect(db_path)
            _ensure_metadata_tables(_conn)
            _auto_reattach_parquet(_conn)
    return _conn


# ---------------------------------------------------------------------------
# Metadata table bootstrap
# ---------------------------------------------------------------------------

def _ensure_metadata_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all metadata tables if they do not exist.

    These tables are the single source of truth for schema, relationships,
    and data registry. They survive process restarts because the DB is
    file-backed by default.
    """
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

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _column_catalog (
            dataset_name  VARCHAR NOT NULL,
            column_name   VARCHAR NOT NULL,
            column_type   VARCHAR,           -- DuckDB type string
            is_metric     BOOLEAN DEFAULT FALSE,
            is_dimension  BOOLEAN DEFAULT FALSE,
            is_join_key   BOOLEAN DEFAULT FALSE,
            description   VARCHAR,
            sample_values VARCHAR,           -- JSON array of up to 5 representative values
            PRIMARY KEY (dataset_name, column_name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _relationships (
            relationship_id  INTEGER PRIMARY KEY,
            left_table       VARCHAR NOT NULL,
            left_column      VARCHAR NOT NULL,
            right_table      VARCHAR NOT NULL,
            right_column     VARCHAR NOT NULL,
            cardinality      VARCHAR DEFAULT 'many-to-one',  -- e.g. many-to-one, one-to-one
            description      VARCHAR,
            added_at         TIMESTAMP DEFAULT current_timestamp,
            UNIQUE (left_table, left_column, right_table, right_column)
        )
    """)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS _rel_seq START 1
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _table_context (
            dataset_name  VARCHAR NOT NULL PRIMARY KEY,
            summary       VARCHAR,        -- one sentence: what this table contains
            grain         VARCHAR,        -- e.g. "one row per order per sales rep"
            tags          VARCHAR,        -- JSON array of keyword tags
            updated_at    TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # dataset_name uses '' as the sentinel for "applies to all tables".
    # DuckDB does not support function expressions (e.g. COALESCE) inside a
    # PRIMARY KEY definition, so we use NOT NULL DEFAULT '' and a UNIQUE
    # constraint instead — semantically identical.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _semantic_map (
            term          VARCHAR NOT NULL,
            dataset_name  VARCHAR NOT NULL DEFAULT '',
            column_name   VARCHAR NOT NULL,
            description   VARCHAR,
            UNIQUE (term, dataset_name)
        )
    """)


# ---------------------------------------------------------------------------
# Parquet auto-reattach
# ---------------------------------------------------------------------------

def _auto_reattach_parquet(conn: duckdb.DuckDBPyConnection) -> None:
    """Re-create views for all registered Parquet files on startup.

    This means analysts never need to re-upload files after a process restart.
    Missing Parquet files are skipped silently (the registry entry is retained
    so the user can re-ingest if needed).
    """
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
    """Introspect a newly loaded table and populate _column_catalog.

    Called automatically by load_file() after the table is created.
    Classifies each column as metric / dimension based on DuckDB type.
    Collects up to 5 sample values for context.

    Existing catalog entries for this table are replaced (re-ingestion).
    """
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

        # Collect sample values (up to 5 distinct non-null)
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
            False,   # is_join_key — set explicitly via register_relationship()
            None,    # description — set by agent or human later
            sample_values,
        ))

    conn.executemany(
        "INSERT INTO _column_catalog "
        "(dataset_name, column_name, column_type, is_metric, is_dimension, is_join_key, description, sample_values) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows_to_insert,
    )


def infer_and_register_relationships(conn: duckdb.DuckDBPyConnection, new_table: str) -> list[dict]:
    """Auto-detect probable join keys between new_table and all existing tables.

    Strategy: find columns with identical names and compatible types across tables.
    Inferred relationships are inserted with cardinality='inferred'.
    Returns a list of dicts describing what was registered.

    NOTE: Inference is best-effort. Use register_relationship() for authoritative joins.
    """
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
                # Type compatibility: both strings, or both numeric
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
                        # Mark the column as a join key in both tables
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
    """Explicitly register an authoritative join relationship.

    This is the definitive version; it overwrites any inferred entry for the
    same (left_table, left_column, right_table, right_column) quad.
    """
    conn.execute(
        """
        INSERT INTO _relationships
            (relationship_id, left_table, left_column, right_table, right_column, cardinality, description)
        VALUES (nextval('_rel_seq'), ?, ?, ?, ?, ?, ?)
        ON CONFLICT (left_table, left_column, right_table, right_column)
        DO UPDATE SET cardinality = excluded.cardinality,
                      description = excluded.description,
                      added_at    = current_timestamp
        """,
        [left_table, left_column, right_table, right_column, cardinality, description],
    )
    # Mark both columns as join keys
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
    """Return a compact schema block for the given tables.

    Used by the sql_writer node to inject only the tables relevant to
    the current query — not the full catalog.
    """
    conn = get_connection()
    lines = []
    for tbl in table_names:
        ctx = conn.execute(
            "SELECT summary, grain FROM _table_context WHERE dataset_name = ?",
            [tbl],
        ).fetchone()
        summary = ctx[0] if ctx and ctx[0] else ""
        grain = ctx[1] if ctx and ctx[1] else ""
        lines.append(f"## Table: {tbl}")
        if summary:
            lines.append(f"Summary: {summary}")
        if grain:
            lines.append(f"Grain: {grain}")
        cols = conn.execute(
            "SELECT column_name, column_type, is_metric, is_dimension, is_join_key, description, sample_values "
            "FROM _column_catalog WHERE dataset_name = ? ORDER BY column_name",
            [tbl],
        ).fetchall()
        for col_name, col_type, is_metric, is_dim, is_jk, desc, samples in cols:
            flags = []
            if is_metric:
                flags.append("metric")
            if is_dim:
                flags.append("dimension")
            if is_jk:
                flags.append("join_key")
            flag_str = ", ".join(flags)
            desc_str = f" — {desc}" if desc else ""
            lines.append(f"  {col_name} ({col_type}) [{flag_str}]{desc_str}")
            if samples and samples != "[]":
                lines.append(f"    samples: {samples}")
        lines.append("")

    # Relationships among the selected tables
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
            lines.append("## Relationships")
            for lt, lc, rt, rc, card in rels:
                lines.append(f"  {lt}.{lc} → {rt}.{rc}  ({card})")

    return "\n".join(lines)


def get_table_summaries() -> str:
    """Return a short listing of all registered tables with their summaries.

    Used by the orchestrator's select_relevant_tables pass.
    """
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
