"""
tests/test_parquet_persistence.py

All tests share a single module-scoped in-memory DuckDB connection.
The connection is created ONCE in _isolated_env and never reset mid-module.
Parquets are written to the session PARQUET_STORE dir (set by conftest.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module-level isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _isolated_env(tmp_path_factory):
    """
    1. Force DUCKDB_PATH=:memory:
    2. Purge agent.* modules ONCE
    3. Warm up a single connection that all tests in this module share
    4. Teardown: close + purge again so the next module starts clean
    """
    # Step 1 — must happen before any agent import
    prev_db = os.environ.get("DUCKDB_PATH")
    os.environ["DUCKDB_PATH"] = ":memory:"

    # Step 2 — fresh import of agent.database with :memory:
    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]

    # Step 3 — create & warm the shared connection
    import agent.database as db_mod
    db_mod._conn = None
    db_mod.get_connection()           # creates the :memory: DB + metadata tables

    yield

    # Step 4 — teardown
    import agent.database as db_mod2  # re-import in case it changed
    if db_mod2._conn is not None:
        try:
            db_mod2._conn.close()
        except Exception:
            pass
    db_mod2._conn = None

    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]

    if prev_db is not None:
        os.environ["DUCKDB_PATH"] = prev_db
    else:
        os.environ.pop("DUCKDB_PATH", None)


# ---------------------------------------------------------------------------
# Helper
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


# ---------------------------------------------------------------------------
# Parquet persistence tests
# All use the shared :memory: connection — never reset _conn between them.
# ---------------------------------------------------------------------------

class TestParquetPersistence:

    def test_parquet_file_created(self, tmp_path):
        from agent.tools import load_file, get_parquet_path
        xl = tmp_path / "sales.xlsx"
        _make_sample_excel(xl)
        result = load_file.invoke({"path": str(xl), "dataset_name": "sales"})
        assert "Successfully loaded" in result, result
        pq = get_parquet_path("sales")
        assert pq.exists(), f"Parquet not found: {pq}"

    def test_parquet_readable_by_pandas(self, tmp_path):
        from agent.tools import load_file, get_parquet_path
        xl = tmp_path / "sales2.xlsx"
        original = _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales2"})
        loaded = pd.read_parquet(get_parquet_path("sales2"))
        assert list(loaded.columns) == list(original.columns)
        assert len(loaded) == len(original)

    def test_duckdb_view_queryable(self, tmp_path):
        """View is queryable on the same shared connection."""
        from agent.tools import load_file
        from agent.database import get_connection
        xl = tmp_path / "sales3.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales3"})
        # get_connection() returns the SAME :memory: instance — no reset
        row = get_connection().execute('SELECT COUNT(*) FROM "sales3"').fetchone()
        assert row[0] == 5

    def test_view_survives_connection_reset(self, tmp_path):
        """With a file-backed DB the view is restored via _auto_reattach_parquet."""
        import agent.database as db_mod
        from agent.tools import load_file, get_parquet_path

        file_db = tmp_path / "restart.duckdb"
        prev_db = os.environ.get("DUCKDB_PATH")
        os.environ["DUCKDB_PATH"] = str(file_db)

        # Give this sub-test its own fresh file DB
        db_mod._conn = None
        for mod in [m for m in sys.modules if m.startswith("agent")]:
            del sys.modules[mod]

        try:
            from agent.tools import load_file as lf  # noqa: F811
            import agent.database as db2

            xl = tmp_path / "sales4.xlsx"
            _make_sample_excel(xl)
            lf.invoke({"path": str(xl), "dataset_name": "sales4"})
            pq = get_parquet_path("sales4")

            # Simulate restart: close and reset
            if db2._conn:
                db2._conn.close()
            db2._conn = None
            for mod in [m for m in sys.modules if m.startswith("agent")]:
                del sys.modules[mod]

            from agent.database import get_connection as gc
            conn = gc()   # _auto_reattach_parquet runs here
            row = conn.execute('SELECT COUNT(*) FROM "sales4"').fetchone()
            assert row[0] == 5

        finally:
            try:
                import agent.database as db_fin
                if db_fin._conn:
                    db_fin._conn.close()
                db_fin._conn = None
            except Exception:
                pass
            for mod in [m for m in sys.modules if m.startswith("agent")]:
                del sys.modules[mod]
            # Restore :memory: for remaining tests in this module
            os.environ["DUCKDB_PATH"] = ":memory:"
            if prev_db is not None:
                pass   # leave :memory: — remaining tests need it
            import agent.database as db_restore
            db_restore._conn = None
            db_restore.get_connection()   # re-warm :memory:

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
# Column catalog tests
# Load 'demo' once at class scope, query the shared connection.
# ---------------------------------------------------------------------------

class TestColumnCatalog:

    @pytest.fixture(scope="class", autouse=True)
    def _load_demo(self, tmp_path_factory):
        from agent.tools import load_file
        xl = tmp_path_factory.mktemp("demo_data") / "demo.xlsx"
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
        assert json.loads(search_semantic_lookup.invoke({"query": "zzznomatchxxx", "dataset": "demo"})) == []


class TestSchemaLookupToolInSuite:

    def test_search_semantic_lookup_in_sql_writer_tools(self):
        from agent.tools import SQL_WRITER_TOOLS
        assert "search_semantic_lookup" in [t.name for t in SQL_WRITER_TOOLS]
