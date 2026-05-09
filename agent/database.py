"""DuckDB connection management, metadata tables, and Parquet store setup."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import duckdb

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None

# Directory where ingested Parquet files are persisted.
PARQUET_STORE = Path(os.getenv("PARQUET_STORE", "./data/parquet"))


def get_parquet_path(dataset_name: str) -> Path:
    """Return the canonical Parquet file path for a dataset."""
    return PARQUET_STORE / f"{dataset_name}.parquet"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection, creating it on first call."""
    global _conn
    with _lock:
        if _conn is None:
            db_path = os.getenv("DUCKDB_PATH", ":memory:")
            _conn = duckdb.connect(db_path)
            _ensure_metadata_tables(_conn)
            _restore_parquet_views(_conn)
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _semantic_lookup (
            dataset_name  VARCHAR NOT NULL,
            column_name   VARCHAR NOT NULL,
            data_type     VARCHAR NOT NULL,
            sample_values VARCHAR,
            n_distinct    BIGINT,
            null_frac     DOUBLE,
            description   VARCHAR,
            PRIMARY KEY (dataset_name, column_name)
        )
        """
    )


def _restore_parquet_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Re-create DuckDB views for any Parquet files already on disk.

    This makes previously ingested datasets available immediately after a
    server restart without re-uploading the source Excel/CSV.
    """
    if not PARQUET_STORE.exists():
        return
    for pq_file in PARQUET_STORE.glob("*.parquet"):
        dataset_name = pq_file.stem
        try:
            conn.execute(
                f"CREATE OR REPLACE VIEW \"{dataset_name}\" AS "
                f"SELECT * FROM read_parquet('{pq_file.as_posix()}')"
            )
        except Exception:
            pass  # best-effort; if the file is corrupt it won't block startup
