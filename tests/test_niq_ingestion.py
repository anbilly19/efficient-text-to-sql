"""
tests/test_niq_ingestion.py

Regression suite for the NIQ ingestion pipeline:
  load_niq_file → classify_niq_columns → detect_niq_structure
  → _table_context grain/tags → _semantic_map aliases

Environment
-----------
Set NIQ_SYNTHETIC_PATH to the absolute path of a NIQ-shaped Excel/CSV/Parquet
file before running.  The suite is automatically skipped when the variable is
absent so CI does not fail without a binary fixture.

Example
-------
    NIQ_SYNTHETIC_PATH=/path/to/niq_panel_synthetic.xlsx pytest tests/test_niq_ingestion.py -v

Isolation model (mirrors test_parquet_persistence.py)
------------------------------------------------------
- Module fixture _module_env: sets DUCKDB_PATH=:memory: and PARQUET_STORE to a
  temp dir, purges agent.* once, re-imports fresh.
- Per-test fixture _fresh_conn: closes and resets _conn between tests so each
  test starts with a clean in-memory database.
"""
from __future__ import annotations

import json
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Skip sentinel
# ---------------------------------------------------------------------------

NIQ_SYNTHETIC_PATH = os.environ.get("NIQ_SYNTHETIC_PATH", "")

pytestmark = pytest.mark.skipif(
    not NIQ_SYNTHETIC_PATH,
    reason=(
        "NIQ_SYNTHETIC_PATH not set. "
        "Export NIQ_SYNTHETIC_PATH=/path/to/niq_panel_synthetic.xlsx to run."
    ),
)


# ---------------------------------------------------------------------------
# Module-level helpers (same pattern as test_parquet_persistence.py)
# ---------------------------------------------------------------------------

def _db():
    return sys.modules["agent.database"]


def _tools():
    return sys.modules["agent.tools"]


# ---------------------------------------------------------------------------
# Module fixture: purge agent.* once, import fresh, keep alive for module.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _module_env(tmp_path_factory):
    store = tmp_path_factory.mktemp("niq_pq_store")
    prev_store = os.environ.get("PARQUET_STORE")
    prev_db    = os.environ.get("DUCKDB_PATH")

    os.environ["PARQUET_STORE"] = str(store)
    os.environ["DUCKDB_PATH"]   = ":memory:"

    for mod in list(sys.modules):
        if mod.startswith("agent"):
            del sys.modules[mod]

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
# Per-test: reset _conn to a clean :memory: database.
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
    db.get_connection()
    yield


# ---------------------------------------------------------------------------
# Helper: run load_niq_file and assert no ERROR prefix
# ---------------------------------------------------------------------------

def _load(dataset_name: str = "niq_panel") -> str:
    result = _tools().load_niq_file.invoke(
        {"path": NIQ_SYNTHETIC_PATH, "dataset_name": dataset_name}
    )
    assert not result.startswith("ERROR"), f"load_niq_file failed: {result}"
    return result


# ---------------------------------------------------------------------------
# TestNIQRegistry
# ---------------------------------------------------------------------------

class TestNIQRegistry:

    def test_data_registry_contains_niq_panel(self):
        _load()
        row = _db().get_connection().execute(
            "SELECT dataset_name FROM _data_registry WHERE dataset_name = 'niq_panel'"
        ).fetchone()
        assert row is not None, "_data_registry must contain niq_panel after load_niq_file"

    def test_result_string_contains_grain_marker(self):
        result = _load()
        assert "NIQ grain:" in result, (
            f"Expected 'NIQ grain:' in result string, got: {result}"
        )

    def test_result_string_contains_semantic_alias_count(self):
        result = _load()
        assert "Seeded" in result and "semantic aliases" in result, (
            f"Expected semantic alias count in result, got: {result}"
        )


# ---------------------------------------------------------------------------
# TestNIQColumnCatalog
# ---------------------------------------------------------------------------

class TestNIQColumnCatalog:

    @pytest.fixture(autouse=True)
    def _loaded(self):
        _load()

    def test_minimum_metric_count(self):
        count = _db().get_connection().execute(
            "SELECT COUNT(*) FROM _column_catalog "
            "WHERE dataset_name = 'niq_panel' AND is_metric = TRUE"
        ).fetchone()[0]
        assert count >= 3, f"Expected >= 3 metrics in niq_panel, got {count}"

    def test_minimum_dimension_count(self):
        count = _db().get_connection().execute(
            "SELECT COUNT(*) FROM _column_catalog "
            "WHERE dataset_name = 'niq_panel' AND is_dimension = TRUE"
        ).fetchone()[0]
        assert count >= 2, f"Expected >= 2 dimensions in niq_panel, got {count}"

    def test_period_column_is_dimension(self):
        """Any column with 'period' in its name must be flagged as dimension, not metric."""
        rows = _db().get_connection().execute(
            "SELECT column_name, is_metric, is_dimension FROM _column_catalog "
            "WHERE dataset_name = 'niq_panel' AND LOWER(column_name) LIKE '%period%'"
        ).fetchall()
        for col_name, is_metric, is_dim in rows:
            assert not is_metric, f"Column '{col_name}' should not be a metric"
            assert is_dim,        f"Column '{col_name}' should be a dimension"

    def test_descriptions_are_set(self):
        """All columns in niq_panel must have a non-NULL description after classify_niq_columns."""
        rows = _db().get_connection().execute(
            "SELECT column_name, description FROM _column_catalog "
            "WHERE dataset_name = 'niq_panel' AND description IS NULL"
        ).fetchall()
        assert len(rows) == 0, (
            f"These columns are missing a description: {[r[0] for r in rows]}"
        )

    def test_sample_values_are_valid_json(self):
        rows = _db().get_connection().execute(
            "SELECT column_name, sample_values FROM _column_catalog "
            "WHERE dataset_name = 'niq_panel'"
        ).fetchall()
        for col, sv in rows:
            assert isinstance(json.loads(sv or "[]"), list), (
                f"Column '{col}': sample_values is not a valid JSON list"
            )


# ---------------------------------------------------------------------------
# TestNIQTableContext
# ---------------------------------------------------------------------------

class TestNIQTableContext:

    @pytest.fixture(autouse=True)
    def _loaded(self):
        _load()

    def test_grain_is_populated(self):
        row = _db().get_connection().execute(
            "SELECT grain FROM _table_context WHERE dataset_name = 'niq_panel'"
        ).fetchone()
        assert row is not None, "_table_context has no row for niq_panel"
        assert row[0] is not None and row[0] != "unknown", (
            f"Expected a meaningful grain, got: {row[0]}"
        )

    def test_grain_is_annual_variant(self):
        grain = _db().get_connection().execute(
            "SELECT grain FROM _table_context WHERE dataset_name = 'niq_panel'"
        ).fetchone()[0]
        assert grain in ("annual", "annual, CY vs PY"), (
            f"Unexpected grain value: '{grain}'"
        )

    def test_tags_contain_niq(self):
        tags = _db().get_connection().execute(
            "SELECT tags FROM _table_context WHERE dataset_name = 'niq_panel'"
        ).fetchone()[0]
        assert tags is not None and "niq" in tags.lower(), (
            f"Expected 'niq' in tags, got: {tags}"
        )

    def test_summary_is_populated(self):
        summary = _db().get_connection().execute(
            "SELECT summary FROM _table_context WHERE dataset_name = 'niq_panel'"
        ).fetchone()[0]
        assert summary is not None and len(summary) > 10, (
            f"Summary looks too short or empty: '{summary}'"
        )


# ---------------------------------------------------------------------------
# TestNIQSemanticMap
# ---------------------------------------------------------------------------

class TestNIQSemanticMap:

    @pytest.fixture(autouse=True)
    def _loaded(self):
        _load()

    def test_german_alias_kaeuferreichweite(self):
        row = _db().get_connection().execute(
            "SELECT column_name FROM _semantic_map "
            "WHERE term = 'Käuferreichweite' AND dataset_name = 'niq_panel'"
        ).fetchone()
        assert row is not None, "Käuferreichweite alias not found in _semantic_map"
        assert row[0] == "Penetration %", (
            f"Expected column 'Penetration %', got '{row[0]}'"
        )

    def test_german_alias_ausgaben_je_kaeufer(self):
        row = _db().get_connection().execute(
            "SELECT column_name FROM _semantic_map "
            "WHERE term = 'Ausgaben je Käufer' AND dataset_name = 'niq_panel'"
        ).fetchone()
        assert row is not None, "Ausgaben je Käufer alias not found in _semantic_map"
        assert row[0] == "Spend per Buyer"

    def test_yoy_alias_scoped_to_dataset(self):
        row = _db().get_connection().execute(
            "SELECT term FROM _semantic_map "
            "WHERE term = 'yoy' AND dataset_name = 'niq_panel'"
        ).fetchone()
        assert row is not None, "yoy alias not found scoped to niq_panel"

    def test_lookup_semantic_tool_resolves_german_term(self):
        result = json.loads(
            _tools().lookup_semantic.invoke({"term": "Käuferreichweite"})
        )
        assert result["description"] is not None, (
            "lookup_semantic should return a description for Käuferreichweite"
        )

    def test_total_aliases_seeded(self):
        count = _db().get_connection().execute(
            "SELECT COUNT(*) FROM _semantic_map WHERE dataset_name = 'niq_panel'"
        ).fetchone()[0]
        # _NIQ_SEMANTIC_TERMS has 12 entries
        assert count >= 12, f"Expected >= 12 semantic aliases, got {count}"


# ---------------------------------------------------------------------------
# TestNIQMultiDataset
# ---------------------------------------------------------------------------

class TestNIQMultiDataset:
    """Confirm repeated calls with different dataset_names produce isolated rows."""

    def test_two_datasets_have_separate_registry_rows(self):
        _load("niq_panel")
        _load("niq_panel_petfood")
        rows = _db().get_connection().execute(
            "SELECT dataset_name FROM _data_registry "
            "WHERE dataset_name IN ('niq_panel', 'niq_panel_petfood') "
            "ORDER BY dataset_name"
        ).fetchall()
        names = [r[0] for r in rows]
        assert "niq_panel" in names
        assert "niq_panel_petfood" in names

    def test_two_datasets_have_separate_table_context_rows(self):
        _load("niq_panel")
        _load("niq_panel_petfood")
        count = _db().get_connection().execute(
            "SELECT COUNT(*) FROM _table_context "
            "WHERE dataset_name IN ('niq_panel', 'niq_panel_petfood')"
        ).fetchone()[0]
        assert count == 2, f"Expected 2 distinct _table_context rows, got {count}"

    def test_semantic_aliases_are_scoped_per_dataset(self):
        _load("niq_panel")
        _load("niq_panel_petfood")
        count = _db().get_connection().execute(
            "SELECT COUNT(DISTINCT dataset_name) FROM _semantic_map "
            "WHERE dataset_name IN ('niq_panel', 'niq_panel_petfood')"
        ).fetchone()[0]
        assert count == 2, (
            "Semantic aliases must be independently scoped per dataset_name"
        )

    def test_column_catalog_rows_do_not_bleed_across_datasets(self):
        _load("niq_panel")
        _load("niq_panel_petfood")
        for ds in ("niq_panel", "niq_panel_petfood"):
            count = _db().get_connection().execute(
                "SELECT COUNT(*) FROM _column_catalog WHERE dataset_name = ?", [ds]
            ).fetchone()[0]
            assert count > 0, f"_column_catalog has no rows for {ds}"

    def test_upsert_does_not_duplicate_registry_row(self):
        _load("niq_panel")
        _load("niq_panel")  # second call — must upsert, not duplicate
        count = _db().get_connection().execute(
            "SELECT COUNT(*) FROM _data_registry WHERE dataset_name = 'niq_panel'"
        ).fetchone()[0]
        assert count == 1, f"Upsert created duplicate _data_registry rows: {count}"
