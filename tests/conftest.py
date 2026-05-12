"""Session-level test configuration.

This module is loaded by pytest BEFORE any test module is imported,
which is the only safe moment to set environment variables that
agent modules bake in at import time (PARQUET_STORE, DUCKDB_PATH).

Without this, fixtures that set os.environ inside a test module run
*after* Python has already executed the top-level imports of agent.tools
and agent.database -- meaning the module-level

    PARQUET_STORE = Path(os.getenv("PARQUET_STORE", "./data/parquet"))

captures the *old* value and every Parquet path written during the test
points to a different directory than the one the DuckDB view references.

Note for test_multi_table.py
----------------------------
test_multi_table.py does NOT use in-memory DuckDB -- it runs against the
real on-disk database seeded by scripts/seed_multi_table.py.  The
DUCKDB_PATH override is therefore intentionally NOT set here; each test
module that needs isolation (test_parquet_persistence.py) handles that
via its own _isolated_env fixture.
"""
from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _session_env(tmp_path_factory):
    """Set PARQUET_STORE to a temp dir for the full test session and purge
    any already-imported agent modules so they re-import against the patched
    environment.

    DUCKDB_PATH is left alone here so that test_multi_table.py can connect
    to the real seeded database.  test_parquet_persistence.py overrides it
    to ':memory:' in its own _isolated_env fixture.
    """
    parquet_dir = tmp_path_factory.mktemp("parquet_store_session")
    os.environ["PARQUET_STORE"] = str(parquet_dir)

    # Purge agent modules that may have been collected before this fixture ran.
    for mod_name in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod_name]

    yield parquet_dir

    # Teardown: reset the singleton so subsequent imports start fresh.
    try:
        import agent.database as _db
        _db._conn = None
    except Exception:
        pass
