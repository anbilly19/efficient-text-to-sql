"""
tests/test_parquet_persistence.py

Key invariant
-------------
All code in this module accesses the DuckDB connection through the SAME
agent.database module object. We never do `from agent.database import X`
in test bodies -- we always go via the module reference stored in
sys.modules["agent.database"] so that load_file and the assertions share
exactly the same _conn singleton.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_excel(path: Path) -> pd.DataFrame:
    df = pd.DataFrame({
        "order_id":   [1, 2, 3, 4, 5],
        "customer":   ["Alice", "Bob", "Alice", "Carol", "Bob"],
        "revenue":    [120.5, 340.0, 88.75, 210.0, 560.25],
        "order_date": ["2023-01-15", "2023-03-22", "2023-06-10", "2024-01-05", "2024-02-28"],
        "category":   ["A", "B", "A", "C", "B"],
    })
    df.to_excel(path, index=False)
    return df


def _db():
    """Return agent.database module -- always the live object in sys.modules."""
    return sys.modules["agent.database"]


def _tools():
    """Return agent.tools module -- always the live object in sys.modules."""
    return sys.modules["agent.tools"]


# ---------------------------------------------------------------------------
# Module fixture: purge agent.* ONCE, import fresh, leave alive forever.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _module_env(tmp_path_factory):
    store = tmp_path_factory.mktemp("pq_store")
    prev_store = os.environ.get("PARQUET_STORE")
    prev_db    = os.environ.get("DUCKDB_PATH")

    os.environ["PARQUET_STORE"] = str(store)
    os.environ["DUCKDB_PATH"]   = ":memory:"

    # Purge ONCE so agent.* re-imports with our env vars.
    for mod in list(sys.modules):
        if mod.startswith("agent"):
            del sys.modules[mod]

    # Import now -- these module objects stay in sys.modules for the whole run.
    import agent.database  # noqa: F401
    import agent.tools     # noqa: F401

    yield store

    db = _db()
    if db._conn is not None:
        try:
            db._conn.close()
        except Exception:
            pass
        db._conn = None

    if prev_store is not None:
        os.environ["PARQUET_STORE"] = prev_store
    else:
        os.environ.pop("PARQUET_STORE", None)
    if prev_db is not None:
        os.environ["DUCKDB_PATH"] = prev_db
    else:
        os.environ.pop("DUCKDB_PATH", None)


# ---------------------------------------------------------------------------
# Per-test: close + reset _conn on the live module, open a fresh :memory: DB.
# NO sys.modules purge -- that would create a second module object.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_conn(_module_env):
    db = _db()
    if db._conn is not None:
        try:
            db._conn.close()
        except Exception:
            pass
    db._conn = None
    os.environ["DUCKDB_PATH"] = ":memory:"
    db.get_connection()   # warm once; all code in this test will reuse it
    yield


# ---------------------------------------------------------------------------
# TestParquetPersistence
# ---------------------------------------------------------------------------

class TestParquetPersistence:

    def test_parquet_file_created(self, tmp_path):
        tools = _tools()
        xl = tmp_path / "sales.xlsx"
        _make_sample_excel(xl)
        result = tools.load_file.invoke({"path": str(xl), "dataset_name": "sales"})
        assert "Successfully loaded" in result, result
        assert tools.get_parquet_path("sales").exists()

    def test_parquet_readable_by_pandas(self, tmp_path):
        tools = _tools()
        xl = tmp_path / "sales2.xlsx"
        original = _make_sample_excel(xl)
        tools.load_file.invoke({"path": str(xl), "dataset_name": "sales2"})
        loaded = pd.read_parquet(tools.get_parquet_path("sales2"))
        assert list(loaded.columns) == list(original.columns)
        assert len(loaded) == len(original)

    def test_duckdb_view_queryable(self, tmp_path):
        tools = _tools()
        xl = tmp_path / "sales3.xlsx"
        _make_sample_excel(xl)
        tools.load_file.invoke({"path": str(xl), "dataset_name": "sales3"})
        # Use _db().get_connection() -- same module object as load_file used internally
        row = _db().get_connection().execute('SELECT COUNT(*) FROM "sales3"').fetchone()
        assert row[0] == 5

    def test_view_survives_connection_reset(self, tmp_path):
        """Subprocess: fully isolated process, cannot affect shared module."""
        store = os.environ["PARQUET_STORE"]
        script = textwrap.dedent(f"""
            import os, sys
            import pandas as pd
            from pathlib import Path

            store   = Path(r"{tmp_path / 'pq'}")
            store.mkdir()
            db_path = Path(r"{tmp_path}") / "restart.duckdb"
            xl_path = Path(r"{tmp_path}") / "sales4.xlsx"

            os.environ["PARQUET_STORE"] = str(store)
            os.environ["DUCKDB_PATH"]   = str(db_path)

            pd.DataFrame({{
                "order_id":   [1,2,3,4,5],
                "customer":   ["Alice","Bob","Alice","Carol","Bob"],
                "revenue":    [120.5,340.0,88.75,210.0,560.25],
                "order_date": ["2023-01-15","2023-03-22","2023-06-10","2024-01-05","2024-02-28"],
                "category":   ["A","B","A","C","B"],
            }}).to_excel(xl_path, index=False)

            import agent.database as db_mod
            db_mod._conn = None
            from agent.tools import load_file
            r = load_file.invoke({{"path": str(xl_path), "dataset_name": "sales4"}})
            assert "Successfully loaded" in r, r
            assert (store / "sales4.parquet").exists()

            db_mod._conn.close()
            db_mod._conn = None
            from agent.database import get_connection
            conn = get_connection()   # triggers _auto_reattach_parquet
            row = conn.execute('SELECT COUNT(*) FROM "sales4"').fetchone()
            assert row[0] == 5, f"Expected 5, got {{row[0]}}"
            print("OK")
        """)
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert res.returncode == 0, f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "OK" in res.stdout

    def test_date_columns_stored_as_native_type(self, tmp_path):
        tools = _tools()
        xl = tmp_path / "sales5.xlsx"
        _make_sample_excel(xl)
        tools.load_file.invoke({"path": str(xl), "dataset_name": "sales5"})
        schema = pd.read_parquet(tools.get_parquet_path("sales5")).dtypes
        assert str(schema["order_date"]).startswith("datetime"), (
            f"Expected datetime, got: {schema['order_date']}"
        )


# ---------------------------------------------------------------------------
# TestColumnCatalog
# ---------------------------------------------------------------------------

class TestColumnCatalog:

    @pytest.fixture(autouse=True)
    def _load_demo(self, tmp_path):
        xl = tmp_path / "demo.xlsx"
        _make_sample_excel(xl)
        _tools().load_file.invoke({"path": str(xl), "dataset_name": "demo"})

    def test_column_catalog_rows_exist(self):
        rows = _db().get_connection().execute(
            "SELECT column_name FROM _column_catalog WHERE dataset_name = 'demo'"
        ).fetchall()
        assert {r[0] for r in rows} == {"order_id", "customer", "revenue", "order_date", "category"}

    def test_metric_flag_set_for_numeric_column(self):
        row = _db().get_connection().execute(
            "SELECT is_metric FROM _column_catalog "
            "WHERE dataset_name = 'demo' AND column_name = 'revenue'"
        ).fetchone()
        assert row is not None
        assert row[0] is True

    def test_sample_values_is_valid_json(self):
        rows = _db().get_connection().execute(
            "SELECT column_name, sample_values FROM _column_catalog WHERE dataset_name = 'demo'"
        ).fetchall()
        for col, sv in rows:
            assert isinstance(json.loads(sv or "[]"), list), f"{col}: not a JSON list"

    def test_search_semantic_lookup_by_keyword(self):
        results = json.loads(_tools().search_semantic_lookup.invoke({"query": "revenue", "dataset": "demo"}))
        assert len(results) >= 1
        assert any(r["column"] == "revenue" for r in results)

    def test_search_semantic_lookup_no_dataset_filter(self):
        results = json.loads(_tools().search_semantic_lookup.invoke({"query": "customer"}))
        assert len(results) >= 1
        assert all("column" in r and "dataset" in r for r in results)

    def test_search_semantic_lookup_no_match_returns_empty(self):
        assert json.loads(
            _tools().search_semantic_lookup.invoke({"query": "zzznomatchxxx", "dataset": "demo"})
        ) == []


class TestSchemaLookupToolInSuite:

    def test_search_semantic_lookup_in_sql_writer_tools(self):
        assert "search_semantic_lookup" in [t.name for t in _tools().SQL_WRITER_TOOLS]
