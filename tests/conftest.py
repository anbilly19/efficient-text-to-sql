"""Session-level test configuration.

Sets PARQUET_STORE to a temp dir before any agent module is imported so that
test_parquet_persistence.py writes its parquets into an isolated location.

test_multi_table.py connects to the REAL on-disk DuckDB (seeded by
scripts/seed_multi_table.py) and must NOT have its parquets deleted between
runs -- the seed script owns those files.
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _session_env(tmp_path_factory):
    """Set PARQUET_STORE to a session-scoped temp dir.

    We set the env var first, THEN purge agent.* from sys.modules so that
    any already-imported agent module is re-imported against the patched env.

    DUCKDB_PATH is intentionally NOT overridden here so that
    test_multi_table.py can connect to the real seeded database.
    test_parquet_persistence.py overrides it to ':memory:' via its own
    _isolated_env fixture.
    """
    parquet_dir = tmp_path_factory.mktemp("parquet_store_session")
    os.environ["PARQUET_STORE"] = str(parquet_dir)

    # Purge agent modules so they re-import with the new PARQUET_STORE env var.
    for mod_name in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod_name]

    yield parquet_dir

    # ------------------------------------------------------------------
    # Teardown: only clean up the session temp dir we created.
    # Do NOT touch .local/parquet/ -- those belong to the seed script and
    # must persist across test runs so _auto_reattach_parquet can find them.
    # ------------------------------------------------------------------

    try:
        import agent.database as _db
        _db._conn = None
    except Exception:
        pass

    if parquet_dir.exists():
        shutil.rmtree(parquet_dir, ignore_errors=True)

    os.environ.pop("PARQUET_STORE", None)
