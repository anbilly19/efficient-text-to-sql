"""Session-level test configuration.

This module is loaded by pytest BEFORE any test module is imported,
which is the only safe moment to set environment variables that
agent modules bake in at import time (PARQUET_STORE, DUCKDB_PATH).

After the fix in agent/tools.py, PARQUET_STORE is resolved at call-time
via _get_parquet_store(), so the env-var override always works. This file
still sets the env var early for safety and handles teardown cleanup.

Note for test_multi_table.py
----------------------------
test_multi_table.py does NOT use in-memory DuckDB -- it runs against the
real on-disk database seeded by scripts/seed_multi_table.py. The
DUCKDB_PATH override is therefore intentionally NOT set here; each test
module that needs isolation (test_parquet_persistence.py) handles that
via its own _isolated_env fixture.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REAL_PARQUET_STORE = _PROJECT_ROOT / ".local" / "parquet"
# Datasets written by seed_multi_table.py -- cleaned up after the suite.
_SEEDED_DATASETS = ["sales1000", "sales_rep_targets", "product_metrics"]


@pytest.fixture(scope="session", autouse=True)
def _session_env(tmp_path_factory):
    """Set PARQUET_STORE to a temp dir for the full test session and purge
    any already-imported agent modules so they re-import against the patched
    environment.

    DUCKDB_PATH is left alone here so that test_multi_table.py can connect
    to the real seeded database. test_parquet_persistence.py overrides it
    to ':memory:' in its own _isolated_env fixture.
    """
    parquet_dir = tmp_path_factory.mktemp("parquet_store_session")
    os.environ["PARQUET_STORE"] = str(parquet_dir)

    for mod_name in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod_name]

    yield parquet_dir

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    # 1. Reset the DuckDB singleton.
    try:
        import agent.database as _db
        _db._conn = None
    except Exception:
        pass

    # 2. Delete every parquet written to the session temp dir.
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir, ignore_errors=True)

    # 3. Delete parquets written to the REAL .local/parquet store by
    #    test_multi_table.py (which runs against the real seeded DB and
    #    therefore bypasses the PARQUET_STORE env override).
    for ds in _SEEDED_DATASETS:
        pq = _REAL_PARQUET_STORE / f"{ds}.parquet"
        try:
            if pq.exists():
                pq.unlink()
        except Exception as exc:
            print(f"[conftest teardown] could not delete {pq}: {exc}")

    # 4. Remove the env override.
    os.environ.pop("PARQUET_STORE", None)
