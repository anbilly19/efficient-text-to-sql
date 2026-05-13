"""
tests/test_table_selection.py
------------------------------
Unit tests for the keyword-based _select_relevant_tables logic in
agent/nodes/orchestrator.py.

All tests run against an in-memory DuckDB fixture — no LLM, no API key,
no file I/O required.

Run the whole file:
    pytest tests/test_table_selection.py -v

Run a single class:
    pytest tests/test_table_selection.py::TestKeywordScoring -v

Run a single test:
    pytest tests/test_table_selection.py::TestKeywordScoring::test_quota_query_selects_rep_targets -v

Run with stdout:
    pytest tests/test_table_selection.py -v -s
"""
from __future__ import annotations

import duckdb
import pytest


# ---------------------------------------------------------------------------
# Minimal inline re-implementation of the keyword scorer
# (mirrors agent/nodes/orchestrator.py::_select_relevant_tables)
# so this test file has zero imports from the agent package and
# therefore runs without any env vars or DB files present.
# ---------------------------------------------------------------------------

def _score_tables(conn: duckdb.DuckDBPyConnection, query: str) -> list[str]:
    """Pure-DuckDB keyword scorer — mirrors the production implementation."""
    words = [w.lower() for w in query.split() if len(w) > 2]
    if not words:
        return conn.execute("SELECT table_name FROM _table_context").df()["table_name"].tolist()

    like_clauses = " OR ".join(
        f"(LOWER(tc.summary || ' ' || tc.tags || ' ' || tc.table_name) LIKE '%{w}%')"
        for w in words
    )
    sql = f"""
        SELECT tc.table_name
        FROM _table_context tc
        JOIN _data_registry dr ON tc.table_name = dr.dataset_name
        WHERE {like_clauses}
    """
    rows = conn.execute(sql).fetchall()
    selected = [r[0] for r in rows]
    if not selected:
        selected = conn.execute(
            "SELECT table_name FROM _table_context"
        ).df()["table_name"].tolist()
    return selected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DuckDB pre-loaded with the three seed tables."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE _table_context (
            table_name VARCHAR PRIMARY KEY,
            summary    VARCHAR,
            tags       VARCHAR
        )
    """)
    c.execute("""
        CREATE TABLE _data_registry (
            dataset_name  VARCHAR NOT NULL PRIMARY KEY,
            parquet_path  VARCHAR,
            source_file   VARCHAR,
            row_count     BIGINT,
            column_count  INTEGER
        )
    """)
    c.executemany(
        "INSERT INTO _table_context VALUES (?, ?, ?)",
        [
            (
                "sales1000",
                "Raw sales transactions: order date, sales rep, region, product, category, quantity, unit price, total revenue.",
                "orders transactions revenue region category product",
            ),
            (
                "sales_rep_targets",
                "Per sales-rep per-region quota targets derived from historical revenue. Columns: sales_rep, region, historical_revenue, annual_quota, orders_count, units_sold, avg_order_value.",
                "quota target attainment sales rep region goal",
            ),
            (
                "product_metrics",
                "Aggregated per-product metrics: total revenue, units sold, order count, average unit price per product and category.",
                "product category metrics avg price baseline",
            ),
        ],
    )
    c.executemany(
        "INSERT INTO _data_registry (dataset_name, row_count, column_count) VALUES (?, ?, ?)",
        [("sales1000", 1000, 9), ("sales_rep_targets", 50, 7), ("product_metrics", 30, 6)],
    )
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

class TestKeywordScoring:
    """Run: pytest tests/test_table_selection.py::TestKeywordScoring -v"""

    def test_quota_query_selects_rep_targets(self, conn):
        selected = _score_tables(conn, "Which sales reps are below their quota?")
        assert "sales_rep_targets" in selected

    def test_attainment_query_selects_rep_targets(self, conn):
        selected = _score_tables(conn, "Show me quota attainment by region")
        assert "sales_rep_targets" in selected

    def test_product_query_selects_product_metrics(self, conn):
        selected = _score_tables(conn, "What is the average unit price per product?")
        assert "product_metrics" in selected

    def test_revenue_query_selects_sales1000(self, conn):
        selected = _score_tables(conn, "What is the total revenue for each region?")
        assert "sales1000" in selected

    def test_transactions_query_selects_sales1000(self, conn):
        selected = _score_tables(conn, "How many orders were placed in 2024?")
        assert "sales1000" in selected

    def test_multi_table_query_selects_both(self, conn):
        """A query mentioning quota AND product should hit both dimension tables."""
        selected = _score_tables(
            conn,
            "For each sales rep, show quota attainment broken down by product category",
        )
        assert "sales_rep_targets" in selected
        assert "product_metrics" in selected

    def test_fact_table_included_for_join_queries(self, conn):
        """Quota queries should also pull in the fact table."""
        selected = _score_tables(
            conn,
            "Which reps are below quota? Show their actual revenue and goal.",
        )
        # 'revenue' and 'actual' are in sales1000 tags/summary
        assert "sales1000" in selected


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    """Run: pytest tests/test_table_selection.py::TestFallback -v"""

    def test_empty_query_returns_all_tables(self, conn):
        """An empty / very short query returns all registered tables."""
        selected = _score_tables(conn, "")
        assert set(selected) == {"sales1000", "sales_rep_targets", "product_metrics"}

    def test_no_match_falls_back_to_all_tables(self, conn):
        """A query with zero keyword hits returns all tables (safe fallback)."""
        selected = _score_tables(conn, "xyzzy foobar quux")
        assert set(selected) == {"sales1000", "sales_rep_targets", "product_metrics"}

    def test_short_words_ignored(self, conn):
        """Words ≤2 chars are filtered — query reduces to nothing → all tables returned."""
        selected = _score_tables(conn, "ok go do it")
        assert len(selected) == 3


# ---------------------------------------------------------------------------
# Scoping: results must not include unregistered tables
# ---------------------------------------------------------------------------

class TestScoping:
    """Run: pytest tests/test_table_selection.py::TestScoping -v"""

    def test_only_registered_tables_returned(self, conn):
        selected = _score_tables(conn, "revenue quota product category region")
        known = {"sales1000", "sales_rep_targets", "product_metrics"}
        assert set(selected).issubset(known), f"Unknown tables in result: {set(selected) - known}"

    def test_category_query_returns_subset(self, conn):
        """A very specific query should select fewer than all three tables."""
        selected = _score_tables(conn, "average baseline price metrics")
        # Only product_metrics matches strongly; sales1000 may also match on 'category'
        assert "product_metrics" in selected
        assert "sales_rep_targets" not in selected
