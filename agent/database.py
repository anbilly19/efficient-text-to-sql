"""DuckDB connection management and metadata table setup."""
from __future__ import annotations

import os
import threading

import duckdb

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    global _conn
    with _lock:
        if _conn is None:
            db_path = os.getenv("DUCKDB_PATH")
            if not db_path:
                os.makedirs("./.local/duckdb", exist_ok=True)
                db_path = "./.local/duckdb/efficient-text-to-sql.duckdb"
            _conn = duckdb.connect(db_path)
            _ensure_metadata_tables(_conn)
            _maybe_seed_semantic_map(_conn)
    return _conn


def _ensure_metadata_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create internal metadata tables if they do not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _schema_catalog (
            dataset_name  VARCHAR NOT NULL,
            column_name   VARCHAR NOT NULL,
            data_type     VARCHAR NOT NULL,
            nullable      BOOLEAN,
            description   VARCHAR,
            PRIMARY KEY (dataset_name, column_name)
        )
        """
    )
    # DuckDB does not support COALESCE() inside PRIMARY KEY.
    # Use a plain surrogate PK + UNIQUE index instead.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _semantic_map (
            id           INTEGER PRIMARY KEY,
            term         VARCHAR NOT NULL,
            dataset_name VARCHAR,
            column_name  VARCHAR NOT NULL,
            description  VARCHAR
        )
        """
    )
    # Unique index enforces the (term, dataset_name) business key.
    # CREATE INDEX IF NOT EXISTS is supported in DuckDB >= 0.9.
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_semantic_map_term_dataset
        ON _semantic_map (term, COALESCE(dataset_name, ''))
        """
    )


def _maybe_seed_semantic_map(conn: duckdb.DuckDBPyConnection) -> None:
    """Populate _semantic_map from the static NIQ catalog on cold start."""
    try:
        count = conn.execute("SELECT COUNT(*) FROM _semantic_map").fetchone()[0]
    except Exception:
        return

    if count > 0:
        return

    try:
        from agent.schema_lookup import SCHEMA_CATALOG
    except Exception:
        return

    dataset_name = "NIQ_Haushaltspaneldaten"
    rows: list[tuple] = []

    for col_name, meta in SCHEMA_CATALOG.items():
        desc = meta.get("description")
        rows.append((col_name, dataset_name, col_name, desc))
        en_label = meta.get("en", "")
        if en_label and en_label.lower() != col_name.lower():
            rows.append((en_label, dataset_name, col_name, desc))
        for alias in meta.get("aliases", []):
            if alias and alias.lower() != col_name.lower():
                rows.append((alias, dataset_name, col_name, desc))

    seen: set[tuple] = set()
    deduped: list[tuple] = []
    for row in rows:
        key = (row[0].lower(), row[1])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    # Generate surrogate IDs starting from 1
    try:
        next_id = (conn.execute("SELECT COALESCE(MAX(id), 0) FROM _semantic_map").fetchone()[0] or 0) + 1
        rows_with_id = [(next_id + i, *r) for i, r in enumerate(deduped)]
        conn.executemany(
            """
            INSERT OR IGNORE INTO _semantic_map(id, term, dataset_name, column_name, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows_with_id,
        )
    except Exception:
        return


def resolve_semantic_term(term: str, dataset_name: str | None = None) -> str | None:
    term = term.strip().lower()
    if not term:
        return None

    conn = get_connection()

    if dataset_name:
        row = conn.execute(
            """
            SELECT column_name
            FROM _semantic_map
            WHERE lower(term) = ?
              AND (dataset_name = ? OR dataset_name IS NULL)
            LIMIT 1
            """,
            [term, dataset_name],
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT column_name
            FROM _semantic_map
            WHERE lower(term) = ?
            LIMIT 1
            """,
            [term],
        ).fetchone()

    return row[0] if row else None


def get_semantic_context(table_name: str | None = None) -> str:
    conn = get_connection()

    if table_name:
        rows = conn.execute(
            """
            SELECT term, column_name, COALESCE(description, '')
            FROM _semantic_map
            WHERE dataset_name = ? OR dataset_name IS NULL
            ORDER BY column_name, term
            """,
            [table_name],
        ).fetchall()
        header = f"=== SEMANTIC COLUMN GUIDE — {table_name} ==="
    else:
        rows = conn.execute(
            """
            SELECT COALESCE(dataset_name, '(all)'), term, column_name, COALESCE(description, '')
            FROM _semantic_map
            ORDER BY dataset_name, column_name, term
            """
        ).fetchall()
        header = "=== SEMANTIC COLUMN GUIDE — all datasets ==="

    if not rows:
        return header + "\n(no semantic mappings defined yet)\n"

    from collections import defaultdict
    col_aliases: dict[str, list[str]] = defaultdict(list)
    col_desc: dict[str, str] = {}

    if table_name:
        for term, col, desc in rows:
            col_aliases[col].append(term)
            if desc and col not in col_desc:
                col_desc[col] = desc
    else:
        for ds, term, col, desc in rows:
            key = f"[{ds}] {col}"
            col_aliases[key].append(term)
            if desc and key not in col_desc:
                col_desc[key] = desc

    lines: list[str] = [header]
    for col, aliases in col_aliases.items():
        aliases_str = ", ".join(f'"{ a}"' for a in aliases[:6])
        desc = col_desc.get(col, "")
        lines.append(f'  "{col}" — {desc}')
        lines.append(f'  → user may say: {aliases_str}')
    lines.append("")
    return "\n".join(lines)
