"""Session-level test configuration.

Sets PARQUET_STORE to a temp dir so that test_parquet_persistence.py writes
its parquets into an isolated location that is cleaned up after the session.

IMPORTANT
---------
We do NOT purge agent.* from sys.modules here. Doing so resets _conn to None
and forces a fresh get_connection() call during the test session, which (in
combination with PARQUET_STORE pointing at the temp dir) would cause
load_file to write the seed parquets into the temp dir -- then teardown
deletes them and the next run breaks.

Instead we rely on _get_parquet_store() reading the env var at call-time
(fixed in agent/tools.py) so the override is always respected without
needing a module purge.

test_multi_table.py has its own module-scoped fixture that resets _conn so
it reconnects to the real on-disk DB without disturbing the parquet paths
stored in _data_registry.
"""
from __future__ import annotations

import os
import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def _session_env(tmp_path_factory):
    """Point PARQUET_STORE at a throwaway temp dir for the whole session.

    test_parquet_persistence.py picks this up via _get_parquet_store() and
    writes its own parquets there. test_multi_table.py uses the real DB whose
    views point at .local/parquet/ (written by the seed script).
    """
    parquet_dir = tmp_path_factory.mktemp("parquet_store_session")
    os.environ["PARQUET_STORE"] = str(parquet_dir)

    yield parquet_dir

    # Teardown: delete only the session temp dir. Never touch .local/parquet/.
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir, ignore_errors=True)

    os.environ.pop("PARQUET_STORE", None)
