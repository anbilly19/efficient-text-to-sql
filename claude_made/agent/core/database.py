"""
Persistent DuckDB layer.

All metadata lives in internal underscore-prefixed tables.
No domain knowledge here — this module is a pure storage and query layer.
"""
import duckdb
from pathlib import Path
from typing import Any

_DB_PATH = Path("analytics.duckdb")
_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = duckdb.connect(str(_DB_PATH))
        _bootstrap_metadata_tables(_conn)
    return _conn


def _bootstrap_metadata_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create all internal metadata tables if they don't exist.
    Every table is domain-agnostic — structure is the same regardless of
    what files are loaded.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _data_registry (
            table_name      VARCHAR PRIMARY KEY,
            source_file     VARCHAR,
            row_count       BIGINT,
            col_count       INTEGER,
            loaded_at       TIMESTAMP DEFAULT now()
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _column_catalog (
            table_name      VARCHAR,
            column_name     VARCHAR,
            dtype           VARCHAR,
            null_rate       DOUBLE,
            n_unique        BIGINT,
            min_val         VARCHAR,
            max_val         VARCHAR,
            mean_val        DOUBLE,
            std_val         DOUBLE,
            top_values      VARCHAR,       -- JSON array of top-5 frequent values
            column_role     VARCHAR,       -- dimension | metric | yoy_delta | prior_period | period
            role_confidence DOUBLE,
            PRIMARY KEY (table_name, column_name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _table_context (
            table_name      VARCHAR PRIMARY KEY,
            grain           VARCHAR,       -- e.g. "annual (52-week rolling)"
            period_column   VARCHAR,
            cy_label        VARCHAR,
            py_label        VARCHAR,
            dimension_cols  VARCHAR,       -- JSON array
            metric_cols     VARCHAR,       -- JSON array
            yoy_cols        VARCHAR,       -- JSON array
            summary         VARCHAR        -- LLM-generated plain-English description
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _semantic_map (
            table_name      VARCHAR,
            alias           VARCHAR,       -- any NL term a user might use
            canonical_col   VARCHAR,       -- exact column name in the table
            description     VARCHAR,       -- what it means in plain language
            alias_lang      VARCHAR,       -- 'en', 'de', 'abbrev', etc.
            PRIMARY KEY (table_name, alias)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _query_rules (
            rule_id         INTEGER,
            table_name      VARCHAR,
            condition_tag   VARCHAR,       -- 'uses_yoy_col' | 'group_query' | 'promo_split' etc.
            condition_sql   VARCHAR,       -- fragment to check in generated SQL (regex or literal)
            inject_fragment VARCHAR,       -- SQL fragment to add if condition fires
            inject_position VARCHAR,       -- 'WHERE' | 'GROUP_BY' | 'HAVING' | 'advisory'
            explanation     VARCHAR,
            PRIMARY KEY (rule_id, table_name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _relationships (
            table_a         VARCHAR,
            col_a           VARCHAR,
            table_b         VARCHAR,
            col_b           VARCHAR,
            confidence      DOUBLE,
            join_type       VARCHAR        -- 'exact_name' | 'value_overlap'
        )
    """)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def run_sql(sql: str, params: list[Any] | None = None) -> list[dict]:
    conn = get_conn()
    rel = conn.execute(sql, params or [])
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, row)) for row in rel.fetchall()]


def execute(sql: str, params: list[Any] | None = None) -> None:
    get_conn().execute(sql, params or [])


def table_exists(name: str) -> bool:
    rows = run_sql(
        "SELECT count(*) as n FROM information_schema.tables WHERE table_name = ?",
        [name],
    )
    return rows[0]["n"] > 0


def get_schema_context(table_name: str) -> dict:
    """Return column catalog + table context for a table as a single dict."""
    cols = run_sql(
        "SELECT * FROM _column_catalog WHERE table_name = ?", [table_name]
    )
    ctx_rows = run_sql(
        "SELECT * FROM _table_context WHERE table_name = ?", [table_name]
    )
    ctx = ctx_rows[0] if ctx_rows else {}
    return {"columns": cols, "context": ctx}


def get_semantic_map(table_name: str) -> list[dict]:
    return run_sql(
        "SELECT alias, canonical_col, description, alias_lang "
        "FROM _semantic_map WHERE table_name = ?",
        [table_name],
    )


def get_query_rules(table_name: str) -> list[dict]:
    return run_sql(
        "SELECT * FROM _query_rules WHERE table_name = ? ORDER BY rule_id",
        [table_name],
    )


def get_all_tables() -> list[str]:
    rows = run_sql("SELECT table_name FROM _data_registry ORDER BY loaded_at")
    return [r["table_name"] for r in rows]


def upsert_semantic_entries(entries: list[dict]) -> None:
    conn = get_conn()
    for e in entries:
        conn.execute("""
            INSERT INTO _semantic_map (table_name, alias, canonical_col, description, alias_lang)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (table_name, alias) DO UPDATE SET
                canonical_col = excluded.canonical_col,
                description   = excluded.description,
                alias_lang    = excluded.alias_lang
        """, [e["table_name"], e["alias"], e["canonical_col"],
              e["description"], e.get("alias_lang", "unknown")])


def upsert_query_rules(rules: list[dict]) -> None:
    conn = get_conn()
    for r in rules:
        conn.execute("""
            INSERT INTO _query_rules
                (rule_id, table_name, condition_tag, condition_sql,
                 inject_fragment, inject_position, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (rule_id, table_name) DO UPDATE SET
                condition_tag   = excluded.condition_tag,
                condition_sql   = excluded.condition_sql,
                inject_fragment = excluded.inject_fragment,
                inject_position = excluded.inject_position,
                explanation     = excluded.explanation
        """, [r["rule_id"], r["table_name"], r["condition_tag"],
              r["condition_sql"], r["inject_fragment"],
              r["inject_position"], r["explanation"]])
