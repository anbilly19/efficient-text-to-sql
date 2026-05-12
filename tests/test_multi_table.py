"""Multi-table integration tests for sales1000 + sales_rep_targets.

Pre-requisites (run once before executing this test suite):

    1. Load sample_sales_1000.xlsx through the agent:
           load_file('data/sample_sales_1000.xlsx', 'sales1000')

    2. Generate and load the targets table:
           python scripts/generate_test_tables.py
           load_file('data/sales_rep_targets.xlsx', 'sales_rep_targets')

The tests assert:
    - Correct relationship inference after ingestion
    - Schema context includes relationship metadata
    - Join queries execute without error
    - No fan-out (result row count <= sales_rep_targets row count)
    - HAVING / filter logic across joined tables
    - Single-table queries still work (regression)
"""
from __future__ import annotations

import json

import pytest

from agent.database import get_connection, get_schema_context
from agent.tools import get_relationships, run_sql


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conn():
    return get_connection()


@pytest.fixture(scope="module", autouse=True)
def require_both_tables(conn):
    """Skip the whole module if either table is missing."""
    registered = {
        r[0]
        for r in conn.execute(
            "SELECT dataset_name FROM _data_registry"
        ).fetchall()
    }
    missing = {"sales1000", "sales_rep_targets"} - registered
    if missing:
        pytest.skip(
            f"Required table(s) not loaded: {missing}.\n"
            "Run scripts/generate_test_tables.py then load both Excel files."
        )


# ---------------------------------------------------------------------------
# Relationship tests
# ---------------------------------------------------------------------------

class TestRelationships:
    def test_relationships_exist(self):
        """At least two relationships inferred between the two tables."""
        result = json.loads(get_relationships.invoke({}))
        rels = result["relationships"]
        assert len(rels) >= 2, f"Expected ≥2 relationships, got: {rels}"

    def test_sales_rep_join_key_registered(self):
        result = json.loads(get_relationships.invoke({}))
        join_pairs = {
            (r["left_column"], r["right_column"])
            for r in result["relationships"]
        }
        assert ("sales_rep", "sales_rep") in join_pairs, (
            "sales_rep join key not registered"
        )

    def test_region_join_key_registered(self):
        result = json.loads(get_relationships.invoke({}))
        join_pairs = {
            (r["left_column"], r["right_column"])
            for r in result["relationships"]
        }
        assert ("region", "region") in join_pairs, (
            "region join key not registered"
        )


# ---------------------------------------------------------------------------
# Schema context tests
# ---------------------------------------------------------------------------

class TestSchemaContext:
    def test_schema_context_includes_relationships(self):
        ctx = get_schema_context(["sales1000", "sales_rep_targets"])
        assert "Relationships" in ctx or "relationships" in ctx.lower()
        assert "sales_rep" in ctx

    def test_schema_context_scoped_to_requested_tables(self):
        """Schema context must not leak columns from unrequested tables."""
        ctx = get_schema_context(["sales1000"])
        # annual_quota only exists in sales_rep_targets
        assert "annual_quota" not in ctx


# ---------------------------------------------------------------------------
# Join query tests
# ---------------------------------------------------------------------------

class TestJoinQueries:
    QUOTA_VS_ACTUAL_SQL = """
        SELECT
            s.sales_rep,
            s.region,
            SUM(s.total_revenue)                              AS actual_revenue,
            t.annual_quota,
            ROUND(SUM(s.total_revenue) / t.annual_quota, 4)  AS attainment
        FROM sales1000 s
        JOIN sales_rep_targets t
          ON s.sales_rep = t.sales_rep
         AND s.region    = t.region
        GROUP BY s.sales_rep, s.region, t.annual_quota
        ORDER BY attainment DESC
    """

    def test_quota_vs_actual_executes(self):
        result = run_sql.invoke({"query": self.QUOTA_VS_ACTUAL_SQL})
        assert not result.startswith("ERROR"), f"Query failed: {result}"

    def test_quota_vs_actual_no_fanout(self, conn):
        """Result rows must not exceed the dimension table row count."""
        result = json.loads(run_sql.invoke({"query": self.QUOTA_VS_ACTUAL_SQL}))
        target_rows = conn.execute(
            "SELECT COUNT(*) FROM sales_rep_targets"
        ).fetchone()[0]
        assert result["metadata"]["row_count"] <= target_rows, (
            f"Fan-out detected: {result['metadata']['row_count']} rows > "
            f"{target_rows} rows in sales_rep_targets"
        )

    def test_quota_vs_actual_attainment_range(self):
        """Attainment values must be positive and finite (no div-by-zero)."""
        result = json.loads(run_sql.invoke({"query": self.QUOTA_VS_ACTUAL_SQL}))
        for row in result["rows"]:
            att = float(row["attainment"])
            assert att > 0, f"Non-positive attainment: {row}"
            assert att < 1000, f"Implausibly high attainment (possible div-by-zero): {row}"

    def test_attainment_is_not_uniform(self):
        """Quota values are independent, so attainment must vary across reps."""
        result = json.loads(run_sql.invoke({"query": self.QUOTA_VS_ACTUAL_SQL}))
        attainments = {float(row["attainment"]) for row in result["rows"]}
        assert len(attainments) > 1, (
            "All attainment values are identical — quota may be derived from actuals"
        )

    def test_below_quota_filter(self):
        sql = """
            SELECT
                t.sales_rep, t.region,
                SUM(s.total_revenue) AS actual,
                t.annual_quota
            FROM sales1000 s
            JOIN sales_rep_targets t
              ON s.sales_rep = t.sales_rep
             AND s.region    = t.region
            GROUP BY t.sales_rep, t.region, t.annual_quota
            HAVING SUM(s.total_revenue) < t.annual_quota
            ORDER BY actual ASC
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Below-quota query failed: {result}"

    def test_top_reps_by_attainment_per_region(self):
        sql = """
            WITH ranked AS (
                SELECT
                    s.sales_rep,
                    s.region,
                    SUM(s.total_revenue)                              AS actual,
                    t.annual_quota,
                    ROUND(SUM(s.total_revenue) / t.annual_quota, 4)  AS attainment,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.region
                        ORDER BY SUM(s.total_revenue) / t.annual_quota DESC
                    ) AS rn
                FROM sales1000 s
                JOIN sales_rep_targets t
                  ON s.sales_rep = t.sales_rep
                 AND s.region    = t.region
                GROUP BY s.sales_rep, s.region, t.annual_quota
            )
            SELECT sales_rep, region, actual, annual_quota, attainment
            FROM ranked
            WHERE rn = 1
            ORDER BY region
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Top-rep query failed: {result}"
        parsed = json.loads(result)
        # One row per region
        regions = {row["region"] for row in parsed["rows"]}
        assert len(regions) >= 1


# ---------------------------------------------------------------------------
# Regression: single-table queries must still work
# ---------------------------------------------------------------------------

class TestSingleTableRegression:
    def test_sales1000_aggregate(self):
        sql = """
            SELECT region, SUM(total_revenue) AS total
            FROM sales1000
            GROUP BY region
            ORDER BY total DESC
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Single-table query failed: {result}"
        parsed = json.loads(result)
        assert parsed["metadata"]["row_count"] >= 1

    def test_sales_rep_targets_standalone(self):
        sql = "SELECT * FROM sales_rep_targets ORDER BY annual_quota DESC LIMIT 5"
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Targets table query failed: {result}"
