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


def _purge_agent():
    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]


def _rewarm_memory():
    """Purge agent.*, set DUCKDB_PATH=:memory:, create fresh connection."""
    os.environ["DUCKDB_PATH"] = ":memory:"
    _purge_agent()
    import agent.database as db_mod
    db_mod._conn = None
    db_mod.get_connection()


# ---------------------------------------------------------------------------
# Module-level isolation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _isolated_env(tmp_path_factory):
    """
    - Force DUCKDB_PATH=:memory: for the whole module.
    - Warm ONE shared connection.
    - Pre-load 'demo' dataset so TestColumnCatalog always finds it,
      even after test_view_survives_connection_reset re-warms the connection.
    """
    prev_db = os.environ.get("DUCKDB_PATH")
    _rewarm_memory()

    # Pre-load demo on the fresh connection
    from agent.tools import load_file
    demo_xl = tmp_path_factory.mktemp("demo_setup") / "demo.xlsx"
    _make_sample_excel(demo_xl)
    load_file.invoke({"path": str(demo_xl), "dataset_name": "demo"})

    yield

    # Teardown
    try:
        import agent.database as db_mod
        if db_mod._conn:
            db_mod._conn.close()
        db_mod._conn = None
    except Exception:
        pass
    _purge_agent()
    if prev_db is not None:
        os.environ["DUCKDB_PATH"] = prev_db
    else:
        os.environ.pop("DUCKDB_PATH", None)


# ---------------------------------------------------------------------------
# Parquet persistence tests
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
        row = get_connection().execute('SELECT COUNT(*) FROM "sales3"').fetchone()
        assert row[0] == 5

    def test_view_survives_connection_reset(self, tmp_path):
        """File-backed DB: view is restored via _auto_reattach_parquet on reconnect."""
        file_db = tmp_path / "restart.duckdb"
        prev_db = os.environ.get("DUCKDB_PATH")
        os.environ["DUCKDB_PATH"] = str(file_db)

        _purge_agent()
        try:
            from agent.tools import load_file as lf, get_parquet_path
            import agent.database as db2
            db2._conn = None
            db2.get_connection()

            xl = tmp_path / "sales4.xlsx"
            _make_sample_excel(xl)
            lf.invoke({"path": str(xl), "dataset_name": "sales4"})

            # Simulate restart
            if db2._conn:
                db2._conn.close()
            db2._conn = None
            _purge_agent()

            from agent.database import get_connection as gc
            conn = gc()   # triggers _auto_reattach_parquet
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
            # Restore shared :memory: connection AND reload demo
            _rewarm_memory()
            from agent.tools import load_file as lf2
            from agent.tools import get_parquet_path as gpp
            demo_pq = gpp("demo")
            if demo_pq.exists():
                # Re-register the view on the fresh :memory: connection
                lf2.invoke({"path": str(demo_pq), "dataset_name": "demo"})

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
# 'demo' is guaranteed to exist: loaded in _isolated_env and reloaded in
# test_view_survives_connection_reset's finally block.
# ---------------------------------------------------------------------------

class TestColumnCatalog:

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
