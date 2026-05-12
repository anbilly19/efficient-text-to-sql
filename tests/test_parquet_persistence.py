"""
tests/test_parquet_persistence.py

Unit tests for Parquet persistence and _column_catalog.
Runs fully offline using a fresh in-memory DuckDB.

Fixture hierarchy
-----------------
conftest.py::_session_env   (session, autouse)
    Sets PARQUET_STORE to a temp dir.

_isolated_env               (module, autouse)
    Overrides DUCKDB_PATH to ':memory:', resets _conn singleton once,
    then leaves it alone for the rest of the module so views persist.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module-level isolation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _isolated_env():
    """Point DUCKDB_PATH at ':memory:' and reset the singleton ONCE.
    All tests in this module share the same in-memory connection so that
    views created by load_file survive across test functions.
    """
    os.environ["DUCKDB_PATH"] = ":memory:"

    # Purge once so agent.database re-imports with :memory:
    for mod_name in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod_name]

    import agent.database as db_module
    db_module._conn = None
    # Warm up the connection NOW so all tests share the same instance.
    db_module.get_connection()

    yield

    import agent.database as db_module  # noqa: F811
    db_module._conn = None
    os.environ.pop("DUCKDB_PATH", None)


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParquetPersistence:

    def test_parquet_file_created(self, tmp_path):
        from agent.tools import load_file, get_parquet_path

        xl = tmp_path / "sales.xlsx"
        _make_sample_excel(xl)
        result = load_file.invoke({"path": str(xl), "dataset_name": "sales"})
        assert "Successfully loaded" in result, f"Unexpected result: {result}"
        pq = get_parquet_path("sales")
        assert pq.exists(), f"Parquet file not found at {pq}"
        assert pq.suffix == ".parquet"

    def test_parquet_readable_by_pandas(self, tmp_path):
        from agent.tools import load_file, get_parquet_path

        xl = tmp_path / "sales2.xlsx"
        original = _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales2"})
        pq = get_parquet_path("sales2")
        loaded = pd.read_parquet(pq)
        assert list(loaded.columns) == list(original.columns)
        assert len(loaded) == len(original)

    def test_duckdb_view_queryable(self, tmp_path):
        """View must be queryable on the SAME connection used by load_file."""
        from agent.tools import load_file
        from agent.database import get_connection

        xl = tmp_path / "sales3.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales3"})

        # Use the same connection -- do NOT reset _conn between calls.
        conn = get_connection()
        row = conn.execute('SELECT COUNT(*) FROM "sales3"').fetchone()
        assert row[0] == 5

    def test_view_survives_connection_reset(self, tmp_path):
        """Simulate restart with a file DB so _data_registry survives reset."""
        import agent.database as db_module

        file_db = tmp_path / "restart_test.duckdb"
        prev_db_path = os.environ.get("DUCKDB_PATH")
        os.environ["DUCKDB_PATH"] = str(file_db)

        db_module._conn = None
        for mod_name in [m for m in sys.modules if m.startswith("agent")]:
            del sys.modules[mod_name]

        try:
            from agent.tools import load_file, get_parquet_path
            import agent.database as db_module  # noqa: F811

            xl = tmp_path / "sales4.xlsx"
            _make_sample_excel(xl)
            load_file.invoke({"path": str(xl), "dataset_name": "sales4"})
            pq_path = get_parquet_path("sales4")

            # Simulate restart
            db_module._conn = None
            for mod_name in [m for m in sys.modules if m.startswith("agent")]:
                del sys.modules[mod_name]

            from agent.database import get_connection
            conn = get_connection()  # triggers _auto_reattach_parquet
            row = conn.execute('SELECT COUNT(*) FROM "sales4"').fetchone()
            assert row[0] == 5, "View not restored after connection reset"

        finally:
            try:
                import agent.database as db_m
                if db_m._conn is not None:
                    db_m._conn.close()
                db_m._conn = None
            except Exception:
                pass
            for f in tmp_path.glob("restart_test.duckdb*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                if pq_path.exists():  # type: ignore[possibly-undefined]
                    pq_path.unlink()
            except Exception:
                pass
            for mod_name in [m for m in sys.modules if m.startswith("agent")]:
                del sys.modules[mod_name]
            os.environ["DUCKDB_PATH"] = prev_db_path if prev_db_path is not None else ":memory:"
            import agent.database as db_m  # noqa: F811
            db_m._conn = None
            db_m.get_connection()  # re-warm the shared :memory: connection

    def test_date_columns_stored_as_native_type(self, tmp_path):
        from agent.tools import load_file, get_parquet_path

        xl = tmp_path / "sales5.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales5"})
        pq = get_parquet_path("sales5")
        schema = pd.read_parquet(pq).dtypes
        assert str(schema["order_date"]).startswith("datetime"), (
            f"Expected datetime dtype for order_date, got: {schema['order_date']}"
        )


class TestColumnCatalog:

    @pytest.fixture(scope="class", autouse=True)
    def _load_dataset(self, tmp_path_factory):
        from agent.tools import load_file
        xl = tmp_path_factory.mktemp("sl_data") / "demo.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "demo"})

    def test_column_catalog_rows_exist(self):
        from agent.database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT column_name FROM _column_catalog WHERE dataset_name = 'demo'"
        ).fetchall()
        col_names = {r[0] for r in rows}
        assert {"order_id", "customer", "revenue", "order_date", "category"} == col_names

    def test_metric_flag_set_for_numeric_column(self):
        from agent.database import get_connection
        conn = get_connection()
        row = conn.execute(
            "SELECT is_metric FROM _column_catalog "
            "WHERE dataset_name = 'demo' AND column_name = 'revenue'"
        ).fetchone()
        assert row is not None
        assert row[0] is True, "revenue should be flagged as a metric"

    def test_sample_values_is_valid_json(self):
        from agent.database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT column_name, sample_values FROM _column_catalog WHERE dataset_name = 'demo'"
        ).fetchall()
        for col, sv in rows:
            assert isinstance(json.loads(sv or "[]"), list), f"{col}: sample_values not a JSON list"

    def test_search_semantic_lookup_by_keyword(self):
        from agent.tools import search_semantic_lookup
        raw = search_semantic_lookup.invoke({"query": "revenue", "dataset": "demo"})
        results = json.loads(raw)
        assert len(results) >= 1
        assert any(r["column"] == "revenue" for r in results)

    def test_search_semantic_lookup_no_dataset_filter(self):
        from agent.tools import search_semantic_lookup
        raw = search_semantic_lookup.invoke({"query": "customer"})
        results = json.loads(raw)
        assert len(results) >= 1
        assert all("column" in r and "dataset" in r for r in results)

    def test_search_semantic_lookup_no_match_returns_empty(self):
        from agent.tools import search_semantic_lookup
        raw = search_semantic_lookup.invoke({"query": "zzznomatchxxx", "dataset": "demo"})
        assert json.loads(raw) == []


class TestSchemaLookupToolInSuite:

    def test_search_semantic_lookup_in_sql_writer_tools(self):
        from agent.tools import SQL_WRITER_TOOLS, search_semantic_lookup
        tool_names = [t.name for t in SQL_WRITER_TOOLS]
        assert "search_semantic_lookup" in tool_names
