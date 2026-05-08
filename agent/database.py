"""DuckDB connection management and metadata table setup."""
from __future__ import annotations

import os
import threading

import duckdb

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection, creating it on first call.

    IMPORTANT:
      - If DUCKDB_PATH is not set, we create a small local file so that internal
        metadata tables (_schema_catalog, _semantic_map) are persisted across
        runs instead of living in an in-memory database.
      - Using DUCKDB_PATH=":memory:" is still supported for ephemeral sessions,
        but any semantic edits will be lost on restart.
    """
    global _conn
    with _lock:
        if _conn is None:
            db_path = os.getenv("DUCKDB_PATH")
            if not db_path:
                # Default to a local file-backed DB for persistence
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _semantic_map (
            term         VARCHAR NOT NULL,
            dataset_name VARCHAR,
            column_name  VARCHAR NOT NULL,
            description  VARCHAR,
            PRIMARY KEY (term, COALESCE(dataset_name, ''))
        )
        """
    )


def _maybe_seed_semantic_map(conn: duckdb.DuckDBPyConnection) -> None:
    """Populate _semantic_map from the static NIQ catalog on cold start.

    Runs ONLY when the table is empty so that later human/agent edits
    (INSERT/UPDATE) are preserved across restarts.
    """
    try:
        count = conn.execute("SELECT COUNT(*) FROM _semantic_map").fetchone()[0]
    except Exception:
        return

    if count > 0:
        return

    # Lazy import to avoid circular deps at module load time
    try:
        from agent.schema_lookup import SCHEMA_CATALOG
    except Exception:
        return

    dataset_name = "NIQ_Haushaltspaneldaten"
    rows: list[tuple] = []

    for col_name, meta in SCHEMA_CATALOG.items():
        desc = meta.get("description")
        # exact column name always maps to itself
        rows.append((col_name, dataset_name, col_name, desc))
        # English label
        en_label = meta.get("en", "")
        if en_label and en_label.lower() != col_name.lower():
            rows.append((en_label, dataset_name, col_name, desc))
        # all declared aliases
        for alias in meta.get("aliases", []):
            if alias and alias.lower() != col_name.lower():
                rows.append((alias, dataset_name, col_name, desc))

    # Deduplicate by (term, dataset_name) before inserting
    seen: set[tuple] = set()
    deduped: list[tuple] = []
    for row in rows:
        key = (row[0].lower(), row[1])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    try:
        conn.executemany(
            """
            INSERT INTO _semantic_map(term, dataset_name, column_name, description)
            VALUES (?, ?, ?, ?)
            """,
            deduped,
        )
    except Exception:
        # Best-effort; if seeding fails we still have a working database
        return


def resolve_semantic_term(term: str, dataset_name: str | None = None) -> str | None:
    """Resolve a natural-language term to an exact column name from _semantic_map.

    Returns None if no match is found.

    Example::

        resolve_semantic_term("basket size")                          # -> "Ausgaben pro Einkaufsakt"
        resolve_semantic_term("frequency", "NIQ_Haushaltspaneldaten") # -> "Einkaufsakte pro Käuferhaushalt"
    """
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
    """Return a compact guide describing column meanings and common aliases.

    Reads from _semantic_map (DuckDB) rather than the static Python dict so
    that analyst/agent edits made at runtime are reflected immediately.

    Inject the return value into orchestrator / sql_writer prompts so the LLM
    can map user phrases like ``"basket size"`` to exact column names.
    """
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

    # Group aliases by column for a compact, readable block
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
        aliases_str = ", ".join(f'"{a}"' for a in aliases[:6])
        desc = col_desc.get(col, "")
        lines.append(f'  "{col}" — {desc}')
        lines.append(f'  → user may say: {aliases_str}')
    lines.append("")
    return "\n".join(lines)
