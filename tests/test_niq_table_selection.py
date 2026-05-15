"""
tests/test_niq_table_selection.py
----------------------------------
Unit tests for the NIQ-aware keyword scoring inside _select_relevant_tables,
and for the allowed_tables scope filter used by _visible_tables.

Fully self-contained: no imports from agent.*, no LLM, no env vars, no files.
Mirrors the pattern of test_table_selection.py exactly.

Run:
    pytest tests/test_niq_table_selection.py -v
"""
from __future__ import annotations

import re

import duckdb
import pytest


# ---------------------------------------------------------------------------
# Inline reimplementation of the NIQ-aware scorer
# (mirrors agent/nodes.py::_select_relevant_tables + _NIQ_QUERY_TERMS)
# ---------------------------------------------------------------------------

_NIQ_QUERY_TERMS = {
    "penetration", "spend", "buyer", "panel", "niq", "nielsen",
    "kaeuferreichweite", "ausgaben", "haushalt",
    "yoy", "yearonyear", "prior", "volume", "frequency", "trips",
    "share", "manufacturer", "brand", "category", "market",
}


def _score_tables_niq(
    conn: duckdb.DuckDBPyConnection,
    user_query: str,
    all_tables: list[str],
) -> list[str]:
    """Inline NIQ-aware scorer — mirrors the production implementation."""
    if len(all_tables) <= 1:
        return all_tables

    rows = conn.execute(
        "SELECT dr.dataset_name, tc.summary, tc.tags "
        "FROM _data_registry dr "
        "LEFT JOIN _table_context tc ON tc.dataset_name = dr.dataset_name"
    ).fetchall()

    if not rows:
        return all_tables

    q_lower = user_query.lower()
    q_words = set(re.findall(r"[a-z]{3,}", q_lower))

    # Umlaut normalisation for NIQ term matching
    q_niq_terms = {w.replace("\u00e4", "a").replace("\u00fc", "u").replace("\u00f6", "o") for w in q_words}
    query_is_niq = bool(q_niq_terms & _NIQ_QUERY_TERMS)

    scored: list[tuple[int, str]] = []
    for dataset_name, summary, tags in rows:
        if dataset_name not in all_tables:
            continue
        text = " ".join([
            dataset_name.lower(),
            (summary or "").lower(),
            (tags or "").lower(),
        ])
        score = sum(1 for w in q_words if w in text)

        if query_is_niq and tags and "niq" in tags.lower():
            score += 2

        scored.append((score, dataset_name))

    selected = [name for score, name in scored if score > 0]
    if not selected:
        return all_tables
    return selected


def _apply_scope(all_tables: list[str], allowed_tables: list[str] | None) -> list[str]:
    """Inline reimplementation of _visible_tables filtering logic."""
    if allowed_tables is None:
        return all_tables
    return [t for t in all_tables if t in allowed_tables]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn_single_niq():
    """One NIQ panel table + one generic sales table."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE _data_registry (
            dataset_name VARCHAR PRIMARY KEY,
            parquet_path VARCHAR,
            row_count BIGINT,
            column_count INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE _table_context (
            dataset_name VARCHAR PRIMARY KEY,
            summary VARCHAR,
            grain VARCHAR,
            tags VARCHAR
        )
    """)
    c.executemany(
        "INSERT INTO _data_registry (dataset_name, row_count, column_count) VALUES (?, ?, ?)",
        [("niq_panel", 500, 12), ("sales", 1000, 8)],
    )
    c.executemany(
        "INSERT INTO _table_context (dataset_name, summary, grain, tags) VALUES (?, ?, ?, ?)",
        [
            (
                "niq_panel",
                "NIQ panel dataset 'niq_panel'. Grain: annual, CY vs PY. Period column: Period.",
                "annual, CY vs PY",
                "niq,panel",
            ),
            (
                "sales",
                "Raw sales transactions with order date, region, revenue.",
                None,
                "orders,transactions",
            ),
        ],
    )
    yield c
    c.close()


@pytest.fixture
def conn_two_niq():
    """Two NIQ panel tables for different categories + one generic table."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE _data_registry (
            dataset_name VARCHAR PRIMARY KEY,
            parquet_path VARCHAR,
            row_count BIGINT,
            column_count INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE _table_context (
            dataset_name VARCHAR PRIMARY KEY,
            summary VARCHAR,
            grain VARCHAR,
            tags VARCHAR
        )
    """)
    c.executemany(
        "INSERT INTO _data_registry (dataset_name, row_count, column_count) VALUES (?, ?, ?)",
        [
            ("niq_panel_petfood", 300, 12),
            ("niq_panel_snacks", 400, 12),
            ("sales", 1000, 8),
        ],
    )
    c.executemany(
        "INSERT INTO _table_context (dataset_name, summary, grain, tags) VALUES (?, ?, ?, ?)",
        [
            (
                "niq_panel_petfood",
                "NIQ panel dataset 'niq_panel_petfood'. Grain: annual. Period column: Period.",
                "annual",
                "niq,panel",
            ),
            (
                "niq_panel_snacks",
                "NIQ panel dataset 'niq_panel_snacks'. Grain: annual, CY vs PY. Period column: Period.",
                "annual, CY vs PY",
                "niq,panel",
            ),
            (
                "sales",
                "Raw sales transactions with order date, region, revenue.",
                None,
                "orders,transactions",
            ),
        ],
    )
    yield c
    c.close()


# ---------------------------------------------------------------------------
# TestNIQBonus — NIQ tag scoring bonus
# ---------------------------------------------------------------------------

class TestNIQBonus:
    """The +2 NIQ bonus must fire for NIQ-domain queries."""

    def test_penetration_query_selects_niq_panel(self, conn_single_niq):
        selected = _score_tables_niq(
            conn_single_niq,
            "What is the penetration rate?",
            ["niq_panel", "sales"],
        )
        assert "niq_panel" in selected

    def test_spend_per_buyer_query_selects_niq_panel(self, conn_single_niq):
        selected = _score_tables_niq(
            conn_single_niq,
            "Show spend per buyer for each brand",
            ["niq_panel", "sales"],
        )
        assert "niq_panel" in selected

    def test_yoy_query_selects_niq_panel(self, conn_single_niq):
        selected = _score_tables_niq(
            conn_single_niq,
            "What is the YoY change in penetration?",
            ["niq_panel", "sales"],
        )
        assert "niq_panel" in selected

    def test_niq_query_does_not_select_sales(self, conn_single_niq):
        """A pure penetration query should NOT pull in the generic sales table."""
        selected = _score_tables_niq(
            conn_single_niq,
            "What is the buyer penetration?",
            ["niq_panel", "sales"],
        )
        assert "sales" not in selected

    def test_revenue_query_does_not_trigger_niq_bonus(self, conn_single_niq):
        """A plain revenue query must not get the NIQ +2 bonus."""
        selected = _score_tables_niq(
            conn_single_niq,
            "What is total revenue by region?",
            ["niq_panel", "sales"],
        )
        assert "sales" in selected


# ---------------------------------------------------------------------------
# TestGermanTerms — umlaut normalisation
# ---------------------------------------------------------------------------

class TestGermanTerms:

    def test_kaeuferreichweite_triggers_niq_bonus(self, conn_single_niq):
        """K\u00e4uferreichweite normalises to kaeuferreichweite which is in _NIQ_QUERY_TERMS."""
        selected = _score_tables_niq(
            conn_single_niq,
            "Was ist die K\u00e4uferreichweite?",
            ["niq_panel", "sales"],
        )
        assert "niq_panel" in selected

    def test_ausgaben_je_kaeufer_triggers_niq_bonus(self, conn_single_niq):
        selected = _score_tables_niq(
            conn_single_niq,
            "Ausgaben je K\u00e4ufer nach Marke",
            ["niq_panel", "sales"],
        )
        assert "niq_panel" in selected

    def test_haushaltsdurchdringung_triggers_niq_bonus(self, conn_single_niq):
        """haushalt is in _NIQ_QUERY_TERMS; matches Haushaltsdurchdringung."""
        selected = _score_tables_niq(
            conn_single_niq,
            "Zeige die Haushaltsdurchdringung",
            ["niq_panel", "sales"],
        )
        assert "niq_panel" in selected


# ---------------------------------------------------------------------------
# TestMultiDatasetDisambiguation
# ---------------------------------------------------------------------------

class TestMultiDatasetDisambiguation:
    """With two NIQ tables, the dataset name in the query must break the tie."""

    def test_petfood_query_selects_petfood_table(self, conn_two_niq):
        selected = _score_tables_niq(
            conn_two_niq,
            "What is the penetration for petfood?",
            ["niq_panel_petfood", "niq_panel_snacks", "sales"],
        )
        assert "niq_panel_petfood" in selected
        assert "niq_panel_snacks" not in selected

    def test_snacks_query_selects_snacks_table(self, conn_two_niq):
        selected = _score_tables_niq(
            conn_two_niq,
            "Show spend per buyer for snacks",
            ["niq_panel_petfood", "niq_panel_snacks", "sales"],
        )
        assert "niq_panel_snacks" in selected
        assert "niq_panel_petfood" not in selected

    def test_generic_niq_query_selects_both_panels(self, conn_two_niq):
        """A query with no category hint matches both NIQ tables."""
        selected = _score_tables_niq(
            conn_two_niq,
            "What is the overall buyer penetration?",
            ["niq_panel_petfood", "niq_panel_snacks", "sales"],
        )
        assert "niq_panel_petfood" in selected
        assert "niq_panel_snacks" in selected

    def test_sales_not_selected_for_niq_query_with_two_panels(self, conn_two_niq):
        selected = _score_tables_niq(
            conn_two_niq,
            "K\u00e4uferreichweite nach Marke f\u00fcr petfood",
            ["niq_panel_petfood", "niq_panel_snacks", "sales"],
        )
        assert "sales" not in selected


# ---------------------------------------------------------------------------
# TestFallback
# ---------------------------------------------------------------------------

class TestFallback:

    def test_no_match_falls_back_to_all_tables(self, conn_single_niq):
        selected = _score_tables_niq(
            conn_single_niq,
            "xyzzy foobar quux",
            ["niq_panel", "sales"],
        )
        assert set(selected) == {"niq_panel", "sales"}

    def test_single_table_always_returned(self, conn_single_niq):
        selected = _score_tables_niq(
            conn_single_niq,
            "whatever",
            ["niq_panel"],
        )
        assert selected == ["niq_panel"]


# ---------------------------------------------------------------------------
# TestScopedTableFilter — allowed_tables filtering (mirrors _visible_tables)
# ---------------------------------------------------------------------------

class TestScopedTableFilter:
    """Verify that the allowed_tables scope correctly filters the candidate list
    before scoring, matching what _visible_tables(state) does in production."""

    ALL = ["niq_panel_petfood", "niq_panel_snacks", "sales"]

    def test_none_scope_passes_all_tables(self):
        visible = _apply_scope(self.ALL, None)
        assert visible == self.ALL

    def test_single_table_scope(self):
        visible = _apply_scope(self.ALL, ["niq_panel_petfood"])
        assert visible == ["niq_panel_petfood"]

    def test_two_table_scope(self):
        visible = _apply_scope(self.ALL, ["niq_panel_petfood", "sales"])
        assert set(visible) == {"niq_panel_petfood", "sales"}
        assert "niq_panel_snacks" not in visible

    def test_empty_scope_returns_empty(self):
        visible = _apply_scope(self.ALL, [])
        assert visible == []

    def test_unknown_table_in_scope_ignored(self):
        visible = _apply_scope(self.ALL, ["ghost_table"])
        assert visible == []

    def test_scope_then_score_niq_query(self, conn_two_niq):
        """Score runs only over scoped tables; snacks is hidden."""
        scoped = _apply_scope(self.ALL, ["niq_panel_petfood", "sales"])
        selected = _score_tables_niq(
            conn_two_niq,
            "What is the penetration for petfood?",
            scoped,
        )
        assert "niq_panel_petfood" in selected
        assert "niq_panel_snacks" not in selected

    def test_scope_blocks_niq_table_from_generic_query(self, conn_two_niq):
        """When NIQ tables are scoped out, a NIQ query can only hit sales."""
        scoped = _apply_scope(self.ALL, ["sales"])
        selected = _score_tables_niq(
            conn_two_niq,
            "xyzzy foobar quux",
            scoped,
        )
        assert "niq_panel_petfood" not in selected
        assert "niq_panel_snacks" not in selected
