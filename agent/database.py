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
    """Return the singleton DuckDB connection, creating it on first call.

    When DUCKDB_PATH is set to a file path the database is persisted on disk
    and all metadata tables (_schema_catalog, _semantic_map, _semantic_lookup,
    _data_registry) survive server restarts.  The default ':memory:' is kept
    for quick local testing without any env setup.
    """
    global _conn
    with _lock:
        if _conn is None:
            db_path = os.getenv("DUCKDB_PATH", ":memory:")
            # Ensure parent directory exists when a file path is given
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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
    # Registry of every successfully ingested file so views can be
    # re-attached automatically on a cold start even when DUCKDB_PATH
    # is ':memory:' (in that case the table is rebuilt from disk Parquet
    # files by _restore_parquet_views, so the registry is mainly useful
    # for the file-backed path).
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _data_registry (
            dataset_name   VARCHAR PRIMARY KEY,
            parquet_path   VARCHAR NOT NULL,
            source_file    VARCHAR,
            row_count      BIGINT,
            column_count   INTEGER,
            loaded_at      TIMESTAMP DEFAULT current_timestamp
        )
        """
    )


def _restore_parquet_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Re-create DuckDB views for any Parquet files already on disk.

    Strategy:
    1. If the DB is file-backed the _data_registry table already has the
       right paths — use those so renamed/moved files are not silently lost.
    2. Fall back to scanning PARQUET_STORE/*.parquet (covers the in-memory
       case and the very first boot after switching to a file-backed DB).
    """
    restored: set[str] = set()

    # 1. Registry-based restore (file-backed DB)
    try:
        rows = conn.execute(
            "SELECT dataset_name, parquet_path FROM _data_registry"
        ).fetchall()
        for dataset_name, parquet_path in rows:
            pq = Path(parquet_path)
            if not pq.exists():
                continue
            try:
                conn.execute(
                    f"CREATE OR REPLACE VIEW \"{dataset_name}\" AS "
                    f"SELECT * FROM read_parquet('{pq.as_posix()}')"
                )
                restored.add(dataset_name)
            except Exception:
                pass
    except Exception:
        pass  # table may not exist yet on very first boot

    # 2. Scan-based restore (in-memory DB or datasets not yet in registry)
    if not PARQUET_STORE.exists():
        return
    for pq_file in PARQUET_STORE.glob("*.parquet"):
        dataset_name = pq_file.stem
        if dataset_name in restored:
            continue
        try:
            conn.execute(
                f"CREATE OR REPLACE VIEW \"{dataset_name}\" AS "
                f"SELECT * FROM read_parquet('{pq_file.as_posix()}')"
            )
        except Exception:
            pass
