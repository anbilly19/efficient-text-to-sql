"""DuckDB connection management and metadata table setup."""
from __future__ import annotations

import os
import threading

import duckdb

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection, creating it on first call."""
    global _conn
    with _lock:
        if _conn is None:
            db_path = os.getenv("DUCKDB_PATH", ":memory:")
            _conn = duckdb.connect(db_path)
            _ensure_metadata_tables(_conn)
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
            term            VARCHAR PRIMARY KEY,
            definition_sql  VARCHAR NOT NULL,
            description     VARCHAR
        )
        """
    )
