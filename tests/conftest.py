"""Session-level test configuration.

Fixture execution order
-----------------------
_session_env  (session, autouse)
    Sets PARQUET_STORE to a throwaway temp dir for the whole session.
    Never touches DUCKDB_PATH or _conn -- each test module manages its own
    connection lifecycle.

test_parquet_persistence.py::_isolated_env  (module, autouse)
    Sets DUCKDB_PATH=:memory:, purges agent.*, warms ONE shared connection.
    Tears it down cleanly after the module finishes.

test_multi_table.py::_reconnect_real_db  (module, autouse)
    Pops DUCKDB_PATH (falling back to the on-disk DB), resets _conn ONLY
    when DUCKDB_PATH is NOT ':memory:' to avoid stealing the persistence
    module's connection.
"""
from __future__ import annotations

import os
import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def _session_env(tmp_path_factory):
    """Point PARQUET_STORE at a throwaway temp dir for the whole session."""
    parquet_dir = tmp_path_factory.mktemp("parquet_store_session")
    os.environ["PARQUET_STORE"] = str(parquet_dir)

    yield parquet_dir

    if parquet_dir.exists():
        shutil.rmtree(parquet_dir, ignore_errors=True)
    os.environ.pop("PARQUET_STORE", None)
