"""Unit tests for Parquet persistence and _column_catalog (semantic lookup).

These tests run fully offline - no LangGraph server needed.
They import agent code directly and use a fresh in-memory DuckDB
for each test module session.

Fixture hierarchy
-----------------
conftest.py::_session_env   (scope=session, autouse)
    Sets PARQUET_STORE to a temp dir *before* any agent module is imported,
    so module-level constants in agent.tools pick up the temp path correctly.

_isolated_env               (scope=module, autouse)
    Overrides DUCKDB_PATH to ':memory:', resets the DuckDB singleton, and
    purges sys.modules so agent.database/tools re-bind against both patched
    env vars for this module only.
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
    """Override DUCKDB_PATH to in-memory and reset the connection singleton
    so these tests never touch the real on-disk database.
    """
    os.environ["DUCKDB_PATH"] = ":memory:"

    for mod_name in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod_name]

    import agent.database as db_module
    db_module._conn = None

    yield

    import agent.database as db_module  # noqa: F811
    db_module._conn = None
    os.environ.pop("DUCKDB_PATH", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_excel(path: Path) -> pd.DataFrame:
    """Write a tiny sales DataFrame to an Excel file and return it."""
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
    """load_file converts Excel -> Parquet and registers a DuckDB view."""

    def test_parquet_file_created(self, tmp_path):
        from agent.tools import load_file, PARQUET_STORE, get_parquet_path  # noqa: F401

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
        from agent.tools import load_file
        from agent.database import get_connection

        xl = tmp_path / "sales3.xlsx"
        _make_sample_excel(xl)
        load_file.invoke({"path": str(xl), "dataset_name": "sales3"})

        conn = get_connection()
        row = conn.execute('SELECT COUNT(*) FROM "sales3"').fetchone()
        assert row[0] == 5

    def test_view_survives_connection_reset(self, tmp_path):
        """Simulate a server restart: _data_registry persists in a file DB so
        _auto_reattach_parquet can rebuild the view on reconnect.

        This test cannot use :memory: because resetting _conn destroys the
        entire in-memory database including _data_registry. A temp .duckdb
        file is used instead so the registry survives the singleton reset.
        """
        import agent.database as db_module

        # Switch to a temp file DB just for this test.
        file_db = tmp_path / "restart_test.duckdb"
        prev_db_path = os.environ.get("DUCKDB_PATH")
        os.environ["DUCKDB_PATH"] = str(file_db)

        # Purge singleton and modules so they bind against the new path.
        db_module._conn = None
        for mod_name in [m for m in sys.modules if m.startswith("agent")]:
            del sys.modules[mod_name]

        try:
            from agent.tools import load_file
            import agent.database as db_module  # noqa: F811

            xl = tmp_path / "sales4.xlsx"
            _make_sample_excel(xl)
            load_file.invoke({"path": str(xl), "dataset_name": "sales4"})

            # Simulate restart: drop singleton — file DB keeps _data_registry.
            db_module._conn = None
            for mod_name in [m for m in sys.modules if m.startswith("agent")]:
                del sys.modules[mod_name]

            from agent.database import get_connection
            conn = get_connection()  # triggers _auto_reattach_parquet from registry
            row = conn.execute('SELECT COUNT(*) FROM "sales4"').fetchone()
            assert row[0] == 5, "View not restored after connection reset"

        finally:
            # Restore :memory: for all subsequent tests.
            import agent.database as db_m
            db_m._conn = None
            for mod_name in [m for m in sys.modules if m.startswith("agent")]:
                del sys.modules[mod_name]
            if prev_db_path is not None:
                os.environ["DUCKDB_PATH"] = prev_db_path
            else:
                os.environ["DUCKDB_PATH"] = ":memory:"
            import agent.database as db_m  # noqa: F811
            db_m._conn = None

    def test_date_columns_stored_as_native_type(self, tmp_path):
        """order_date (ISO strings in Excel) should land as datetime in Parquet."""
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
    """_column_catalog is populated correctly and search_semantic_lookup works.

    NOTE: This branch replaced _semantic_lookup with _column_catalog.
    n_distinct and null_frac are no longer stored as columns; sample_values
    and description are still present.
    """

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
            parsed = json.loads(sv or "[]")
            assert isinstance(parsed, list), f"{col}: sample_values is not a JSON list"

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
        results = json.loads(raw)
        assert results == []


class TestSchemaLookupToolInSuite:
    """search_semantic_lookup is exposed in SQL_WRITER_TOOLS."""

    def test_search_semantic_lookup_in_sql_writer_tools(self):
        from agent.tools import SQL_WRITER_TOOLS, search_semantic_lookup
        tool_names = [t.name for t in SQL_WRITER_TOOLS]
        assert "search_semantic_lookup" in tool_names
