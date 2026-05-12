"""
tests/test_parquet_persistence.py

Design
------
- _isolated_env (module, autouse): boots ONE shared :memory: connection for
  TestParquetPersistence and TestColumnCatalog. Never torn down mid-module.
- test_view_survives_connection_reset: fully self-contained using a
  subprocess so it cannot corrupt the shared connection.
- TestColumnCatalog: each test loads its own data into the shared connection.
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


# ---------------------------------------------------------------------------
# Module isolation: one shared :memory: connection for this whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _isolated_env():
    prev_db = os.environ.get("DUCKDB_PATH")
    os.environ["DUCKDB_PATH"] = ":memory:"

    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]

    import agent.database as db_mod
    db_mod._conn = None
    db_mod.get_connection()   # warm once

    yield

    try:
        import agent.database as db_mod2
        if db_mod2._conn:
            db_mod2._conn.close()
        db_mod2._conn = None
    except Exception:
        pass
    for mod in [m for m in sys.modules if m.startswith("agent")]:
        del sys.modules[mod]
    if prev_db is not None:
        os.environ["DUCKDB_PATH"] = prev_db
    else:
        os.environ.pop("DUCKDB_PATH", None)


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
        row = get_connection().execute('SELECT COUNT(*) FROM "sales3"').fetchone()
        assert row[0] == 5

    def test_view_survives_connection_reset(self, tmp_path):
        """
        Run in a subprocess so it has its own process state and cannot
        disturb the shared :memory: connection used by the rest of this module.
        """
        script = textwrap.dedent(f"""
            import os, sys, pandas as pd
            from pathlib import Path

            parquet_store = Path(r"{os.environ['PARQUET_STORE']}")
            db_path = Path(r"{tmp_path}") / "restart.duckdb"
            xl_path = Path(r"{tmp_path}") / "sales4.xlsx"

            os.environ["PARQUET_STORE"] = str(parquet_store)
            os.environ["DUCKDB_PATH"] = str(db_path)

            # Write excel
            df = pd.DataFrame({{
                "order_id": [1,2,3,4,5],
                "customer": ["Alice","Bob","Alice","Carol","Bob"],
                "revenue": [120.5,340.0,88.75,210.0,560.25],
                "order_date": ["2023-01-15","2023-03-22","2023-06-10","2024-01-05","2024-02-28"],
                "category": ["A","B","A","C","B"],
            }})
            df.to_excel(xl_path, index=False)

            from agent.tools import load_file
            import agent.database as db_mod
            db_mod._conn = None
            result = load_file.invoke({{"path": str(xl_path), "dataset_name": "sales4"}})
            assert "Successfully loaded" in result, result

            # Verify parquet path is absolute and exists
            pq = parquet_store / "sales4.parquet"
            assert pq.exists(), f"Parquet missing: {{pq}}"

            # Simulate restart
            db_mod._conn.close()
            db_mod._conn = None

            from agent.database import get_connection
            conn = get_connection()  # triggers _auto_reattach_parquet
            row = conn.execute('SELECT COUNT(*) FROM "sales4"').fetchone()
            assert row[0] == 5, f"Expected 5, got {{row[0]}}"
            print("OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert "OK" in result.stdout

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
# Each test loads its own dataset to avoid dependency on test ordering.
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
