"""
tests/test_parquet_persistence.py

Completely self-contained: every test receives an explicit DuckDB connection
fixture. No reliance on the agent.database._conn singleton, so other modules
cannot interfere with this module's state regardless of test ordering.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import duckdb
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":   [1, 2, 3, 4, 5],
        "customer":   ["Alice", "Bob", "Alice", "Carol", "Bob"],
        "revenue":    [120.5, 340.0, 88.75, 210.0, 560.25],
        "order_date": ["2023-01-15", "2023-03-22", "2023-06-10", "2024-01-05", "2024-02-28"],
        "category":   ["A", "B", "A", "C", "B"],
    })


def _make_sample_excel(path: Path) -> pd.DataFrame:
    df = _make_sample_df()
    df.to_excel(path, index=False)
    return df


def _fresh_conn(parquet_store: Path) -> duckdb.DuckDBPyConnection:
    """Open a fresh :memory: DuckDB, set PARQUET_STORE, boot agent metadata."""
    os.environ["PARQUET_STORE"] = str(parquet_store)
    os.environ["DUCKDB_PATH"] = ":memory:"

    # Purge cached modules so agent.database re-imports cleanly
    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]

    import agent.database as db_mod
    db_mod._conn = None
    return db_mod.get_connection()


# ---------------------------------------------------------------------------
# Module-level env: just set PARQUET_STORE; each test calls _fresh_conn.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _module_env(tmp_path_factory):
    """Provide a stable parquet store dir for the whole module."""
    store = tmp_path_factory.mktemp("pq_store")
    prev_store = os.environ.get("PARQUET_STORE")
    prev_db    = os.environ.get("DUCKDB_PATH")
    os.environ["PARQUET_STORE"] = str(store)
    os.environ["DUCKDB_PATH"]   = ":memory:"

    # Purge + warm once so imports work
    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]
    import agent.database as db_mod
    db_mod._conn = None
    db_mod.get_connection()

    yield store

    # Restore env
    if prev_store is not None:
        os.environ["PARQUET_STORE"] = prev_store
    else:
        os.environ.pop("PARQUET_STORE", None)
    if prev_db is not None:
        os.environ["DUCKDB_PATH"] = prev_db
    else:
        os.environ.pop("DUCKDB_PATH", None)


@pytest.fixture(autouse=True)
def _reset_conn(_module_env):
    """Before each test: ensure we have a fresh :memory: connection on
    PARQUET_STORE=_module_env so load_file always uses the right DB."""
    os.environ["PARQUET_STORE"] = str(_module_env)
    os.environ["DUCKDB_PATH"]   = ":memory:"
    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]
    import agent.database as db_mod
    db_mod._conn = None
    db_mod.get_connection()
    yield


# ---------------------------------------------------------------------------
# TestParquetPersistence
# ---------------------------------------------------------------------------

class TestParquetPersistence:

    def test_parquet_file_created(self, tmp_path):
        from agent.tools import load_file, get_parquet_path
        xl = tmp_path / "sales.xlsx"
        _make_sample_excel(xl)
        result = load_file.invoke({"path": str(xl), "dataset_name": "sales"})
        assert "Successfully loaded" in result, result
        assert get_parquet_path("sales").exists()

    def test_parquet_readable_by_pandas(self, tmp_path):
        from agent.tools import load_file, get_parquet_path
        xl = tmp_path / "sales2.xlsx"
        original = _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales2"})
        loaded = pd.read_parquet(get_parquet_path("sales2"))
        assert list(loaded.columns) == list(original.columns)
        assert len(loaded) == len(original)

    def test_duckdb_view_queryable(self, tmp_path):
        from agent.tools import load_file
        from agent.database import get_connection
        xl = tmp_path / "sales3.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales3"})
        # Same connection as load_file used -- view must be here
        row = get_connection().execute('SELECT COUNT(*) FROM "sales3"').fetchone()
        assert row[0] == 5

    def test_view_survives_connection_reset(self, tmp_path):
        """Run in subprocess: owns its own process state entirely."""
        script = textwrap.dedent(f"""
            import os, sys
            import pandas as pd
            from pathlib import Path

            store = Path(r"{tmp_path / 'pq'}")
            store.mkdir()
            db_path = Path(r"{tmp_path}") / "restart.duckdb"
            xl_path = Path(r"{tmp_path}") / "sales4.xlsx"

            os.environ["PARQUET_STORE"] = str(store)
            os.environ["DUCKDB_PATH"]   = str(db_path)

            df = pd.DataFrame({{
                "order_id": [1,2,3,4,5],
                "customer": ["Alice","Bob","Alice","Carol","Bob"],
                "revenue":  [120.5,340.0,88.75,210.0,560.25],
                "order_date": ["2023-01-15","2023-03-22","2023-06-10","2024-01-05","2024-02-28"],
                "category": ["A","B","A","C","B"],
            }})
            df.to_excel(xl_path, index=False)

            import agent.database as db_mod
            db_mod._conn = None
            from agent.tools import load_file
            r = load_file.invoke({{"path": str(xl_path), "dataset_name": "sales4"}})
            assert "Successfully loaded" in r, r

            # Verify parquet exists
            pq = store / "sales4.parquet"
            assert pq.exists(), f"Parquet missing: {{pq}}"

            # Simulate restart
            db_mod._conn.close()
            db_mod._conn = None

            from agent.database import get_connection
            conn = get_connection()   # _auto_reattach_parquet runs here
            row = conn.execute('SELECT COUNT(*) FROM "sales4"').fetchone()
            assert row[0] == 5, f"Expected 5, got {{row[0]}}"
            print("OK")
        """)
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert res.returncode == 0, f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "OK" in res.stdout

    def test_date_columns_stored_as_native_type(self, tmp_path):
        from agent.tools import load_file, get_parquet_path
        xl = tmp_path / "sales5.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales5"})
        schema = pd.read_parquet(get_parquet_path("sales5")).dtypes
        assert str(schema["order_date"]).startswith("datetime"), (
            f"Expected datetime, got: {schema['order_date']}"
        )


# ---------------------------------------------------------------------------
# TestColumnCatalog
# ---------------------------------------------------------------------------

class TestColumnCatalog:

    @pytest.fixture(autouse=True)
    def _load_demo(self, tmp_path):
        from agent.tools import load_file
        xl = tmp_path / "demo.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "demo"})

    def test_column_catalog_rows_exist(self):
        from agent.database import get_connection
        rows = get_connection().execute(
            "SELECT column_name FROM _column_catalog WHERE dataset_name = 'demo'"
        ).fetchall()
        assert {r[0] for r in rows} == {"order_id", "customer", "revenue", "order_date", "category"}

    def test_metric_flag_set_for_numeric_column(self):
        from agent.database import get_connection
        row = get_connection().execute(
            "SELECT is_metric FROM _column_catalog "
            "WHERE dataset_name = 'demo' AND column_name = 'revenue'"
        ).fetchone()
        assert row is not None
        assert row[0] is True

    def test_sample_values_is_valid_json(self):
        from agent.database import get_connection
        rows = get_connection().execute(
            "SELECT column_name, sample_values FROM _column_catalog WHERE dataset_name = 'demo'"
        ).fetchall()
        for col, sv in rows:
            assert isinstance(json.loads(sv or "[]"), list), f"{col}: not a JSON list"

    def test_search_semantic_lookup_by_keyword(self):
        from agent.tools import search_semantic_lookup
        results = json.loads(search_semantic_lookup.invoke({"query": "revenue", "dataset": "demo"}))
        assert len(results) >= 1
        assert any(r["column"] == "revenue" for r in results)

    def test_search_semantic_lookup_no_dataset_filter(self):
        from agent.tools import search_semantic_lookup
        results = json.loads(search_semantic_lookup.invoke({"query": "customer"}))
        assert len(results) >= 1
        assert all("column" in r and "dataset" in r for r in results)

    def test_search_semantic_lookup_no_match_returns_empty(self):
        from agent.tools import search_semantic_lookup
        assert json.loads(
            search_semantic_lookup.invoke({"query": "zzznomatchxxx", "dataset": "demo"})
        ) == []


class TestSchemaLookupToolInSuite:

    def test_search_semantic_lookup_in_sql_writer_tools(self):
        from agent.tools import SQL_WRITER_TOOLS
        assert "search_semantic_lookup" in [t.name for t in SQL_WRITER_TOOLS]
