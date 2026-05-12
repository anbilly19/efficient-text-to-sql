"""
tests/test_multi_table.py
--------------------------
Integration tests for multi-table queries.

Pre-requisites (run once):
    python scripts/generate_test_data.py   # writes data/sales_rep_targets.xlsx + product_metrics.xlsx
    python scripts/seed_multi_table.py     # loads them into DuckDB, registers joins

Run the whole file:
    pytest tests/test_multi_table.py -v
"""
from __future__ import annotations

import json

import pytest

from agent.database import get_connection, get_schema_context
from agent.tools import run_sql
from agent.db.catalog import list_relationships


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conn():
    return get_connection()


@pytest.fixture(scope="module", autouse=True)
def require_tables(conn):
    """Skip the whole module if seed tables are missing."""
    registered = {
        r[0]
        for r in conn.execute(
            "SELECT dataset_name FROM _data_registry"
        ).fetchall()
    }
    missing = {"sales1000", "sales_rep_targets", "product_metrics"} - registered
    if missing:
        pytest.skip(
            f"Required table(s) not loaded: {missing}.\n"
            "Run: python scripts/generate_test_data.py && python scripts/seed_multi_table.py"
        )


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestRelationships:
    """Verify the _relationships registry is correctly populated."""

    def test_at_least_four_relationships(self):
        rels = list_relationships()
        assert len(rels) >= 4, f"Expected >= 4 relationships, got: {rels}"

    def test_sales_rep_join_key_registered(self):
        rels = list_relationships()
        # list_relationships returns dicts with keys left_column / right_column
        # The fact-table column is "Sales Rep" (quoted); the dim column is sales_rep
        pairs = {(r["left_column"], r["right_column"]) for r in rels}
        assert ("Sales Rep", "sales_rep") in pairs or ("sales_rep", "Sales Rep") in pairs, (
            f"sales_rep join key missing. Registered pairs: {pairs}"
        )

    def test_region_join_key_registered(self):
        rels = list_relationships()
        pairs = {(r["left_column"], r["right_column"]) for r in rels}
        assert ("Region", "region") in pairs or ("region", "Region") in pairs, (
            f"region join key missing. Registered pairs: {pairs}"
        )

    def test_product_join_key_registered(self):
        rels = list_relationships()
        # Accept either direction and any casing
        all_cols = {
            c.lower()
            for r in rels
            for c in (r["left_column"], r["right_column"])
        }
        assert any("product" in c for c in all_cols), (
            f"product join key missing. All join columns: {all_cols}"
        )

    def test_category_join_key_registered(self):
        rels = list_relationships()
        all_cols = {
            c.lower()
            for r in rels
            for c in (r["left_column"], r["right_column"])
        }
        assert any("category" in c for c in all_cols), (
            f"category join key missing. All join columns: {all_cols}"
        )

    def test_relationships_scoped_to_tables(self):
        """list_relationships(tables=[...]) returns only relevant rows."""
        rels = list_relationships(tables=["sales1000", "sales_rep_targets"])
        tables_seen = {r["left_table"] for r in rels} | {r["right_table"] for r in rels}
        assert "product_metrics" not in tables_seen, (
            "product_metrics leaked into a sales_rep_targets-scoped relationship query"
        )


# ---------------------------------------------------------------------------
# Schema context
# ---------------------------------------------------------------------------

class TestSchemaContext:
    """Verify schema context is correctly scoped per query."""

    def test_includes_relationships_block(self):
        ctx = get_schema_context(["sales1000", "sales_rep_targets"])
        assert "Relationships" in ctx or "relationships" in ctx.lower(), (
            "Schema context must include a Relationships block"
        )

    def test_scoped_to_requested_tables(self):
        """annual_quota / quota_usd only exist in sales_rep_targets."""
        ctx = get_schema_context(["sales1000"])
        assert "quota" not in ctx.lower(), (
            "Quota column from sales_rep_targets leaked into sales1000-only context"
        )

    def test_product_metrics_columns_present(self):
        ctx = get_schema_context(["sales1000", "product_metrics"])
        assert "product_metrics" in ctx


# ---------------------------------------------------------------------------
# sales1000 x sales_rep_targets
# (fact columns are Title Case; dim columns are snake_case)
# ---------------------------------------------------------------------------

class TestRepTargetsJoin:
    """Queries joining sales1000 (fact) to sales_rep_targets (dimension).

    Column name reference
    ---------------------
    sales1000 fact-table columns (Title Case, must be double-quoted):
      "Sales Rep", "Region", "Total Revenue"

    sales_rep_targets dim-table columns (snake_case, no quoting needed):
      sales_rep, region, quota_usd
    """

    QUOTA_SQL = """
        SELECT
            s."Sales Rep"                                      AS sales_rep,
            s."Region"                                         AS region,
            SUM(s."Total Revenue")                             AS actual_revenue,
            t.quota_usd,
            ROUND(SUM(s."Total Revenue") / t.quota_usd, 4)    AS attainment
        FROM sales1000 s
        JOIN sales_rep_targets t
          ON s."Sales Rep" = t.sales_rep
         AND s."Region"    = t.region
        GROUP BY s."Sales Rep", s."Region", t.quota_usd
        ORDER BY attainment DESC
    """

    def test_quota_vs_actual_executes(self):
        result = run_sql.invoke({"query": self.QUOTA_SQL})
        assert not result.startswith("ERROR"), f"Query failed: {result}"

    def test_no_fanout(self, conn):
        result = json.loads(run_sql.invoke({"query": self.QUOTA_SQL}))
        target_rows = conn.execute(
            "SELECT COUNT(*) FROM sales_rep_targets"
        ).fetchone()[0]
        assert result["metadata"]["row_count"] <= target_rows, (
            f"Fan-out detected: {result['metadata']['row_count']} > {target_rows}"
        )

    def test_attainment_positive_and_finite(self):
        result = json.loads(run_sql.invoke({"query": self.QUOTA_SQL}))
        for row in result["rows"]:
            att = float(row["attainment"])
            assert 0 < att < 1000, f"Implausible attainment {att}: {row}"

    def test_attainment_varies_across_reps(self):
        result = json.loads(run_sql.invoke({"query": self.QUOTA_SQL}))
        attainments = {float(row["attainment"]) for row in result["rows"]}
        assert len(attainments) > 1, (
            "All attainment values identical -- quota may be derived directly from actuals"
        )

    def test_below_quota_filter(self):
        sql = """
            SELECT
                t.sales_rep, t.region,
                SUM(s."Total Revenue") AS actual,
                t.quota_usd
            FROM sales1000 s
            JOIN sales_rep_targets t
              ON s."Sales Rep" = t.sales_rep
             AND s."Region"    = t.region
            GROUP BY t.sales_rep, t.region, t.quota_usd
            HAVING SUM(s."Total Revenue") < t.quota_usd
            ORDER BY actual ASC
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Below-quota query failed: {result}"

    def test_top_rep_per_region(self):
        sql = """
            WITH ranked AS (
                SELECT
                    s."Sales Rep"                                          AS sales_rep,
                    s."Region"                                             AS region,
                    SUM(s."Total Revenue")                                 AS actual,
                    t.quota_usd,
                    ROUND(SUM(s."Total Revenue") / t.quota_usd, 4)        AS attainment,
                    ROW_NUMBER() OVER (
                        PARTITION BY s."Region"
                        ORDER BY SUM(s."Total Revenue") / t.quota_usd DESC
                    ) AS rn
                FROM sales1000 s
                JOIN sales_rep_targets t
                  ON s."Sales Rep" = t.sales_rep
                 AND s."Region"    = t.region
                GROUP BY s."Sales Rep", s."Region", t.quota_usd
            )
            SELECT sales_rep, region, actual, quota_usd, attainment
            FROM ranked WHERE rn = 1
            ORDER BY region
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Top-rep query failed: {result}"
        parsed = json.loads(result)
        regions = {row["region"] for row in parsed["rows"]}
        assert len(regions) >= 1

    def test_quota_gap_ordered(self):
        sql = """
            SELECT
                t.sales_rep,
                t.region,
                t.quota_usd - SUM(s."Total Revenue") AS quota_gap
            FROM sales1000 s
            JOIN sales_rep_targets t
              ON s."Sales Rep" = t.sales_rep
             AND s."Region"    = t.region
            GROUP BY t.sales_rep, t.region, t.quota_usd
            ORDER BY quota_gap DESC
            LIMIT 10
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Quota-gap query failed: {result}"


# ---------------------------------------------------------------------------
# sales1000 x product_metrics
# ---------------------------------------------------------------------------

class TestProductMetricsJoin:
    """Queries joining sales1000 (fact) to product_metrics (dimension).

    Column name reference
    ---------------------
    sales1000 fact-table columns (Title Case, double-quoted):
      "Product Name", "Category", "Total Revenue", "Quantity Ordered", "Unit Price"

    product_metrics dim-table columns (snake_case):
      product_name, category, unit_cost_usd
    """

    def test_revenue_vs_avg_cost(self):
        """Compare line-item revenue against product-level unit cost."""
        sql = """
            SELECT
                s."Product Name",
                s."Category",
                SUM(s."Total Revenue")    AS total_revenue,
                p.unit_cost_usd,
                SUM(s."Quantity Ordered") AS units_sold
            FROM sales1000 s
            JOIN product_metrics p
              ON s."Product Name" = p.product_name
             AND s."Category"     = p.category
            GROUP BY s."Product Name", s."Category", p.unit_cost_usd
            ORDER BY total_revenue DESC
            LIMIT 10
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Query failed: {result}"
        parsed = json.loads(result)
        assert parsed["metadata"]["row_count"] >= 1

    def test_above_avg_cost_products(self):
        """Orders where the unit price paid exceeds the product's sourcing cost."""
        sql = """
            SELECT
                s."Product Name",
                s."Unit Price",
                p.unit_cost_usd,
                s."Unit Price" - p.unit_cost_usd AS cost_margin
            FROM sales1000 s
            JOIN product_metrics p
              ON s."Product Name" = p.product_name
             AND s."Category"     = p.category
            WHERE s."Unit Price" > p.unit_cost_usd
            ORDER BY cost_margin DESC
            LIMIT 10
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Above-cost query failed: {result}"

    def test_product_no_fanout(self, conn):
        sql = """
            SELECT COUNT(*) AS n
            FROM sales1000 s
            LEFT JOIN product_metrics p
              ON s."Product Name" = p.product_name
             AND s."Category"     = p.category
        """
        fact_count = conn.execute("SELECT COUNT(*) FROM sales1000").fetchone()[0]
        result = json.loads(run_sql.invoke({"query": sql}))
        joined_count = result["rows"][0]["n"]
        assert int(joined_count) == fact_count, (
            f"Fan-out: fact={fact_count}, joined={joined_count}"
        )

    def test_top_categories_by_margin_proxy(self):
        """Revenue per unit vs. sourcing cost -- rough margin proxy per category."""
        sql = """
            SELECT
                s."Category",
                ROUND(SUM(s."Total Revenue") / SUM(s."Quantity Ordered"), 2) AS revenue_per_unit,
                AVG(p.unit_cost_usd)                                           AS avg_cost
            FROM sales1000 s
            JOIN product_metrics p
              ON s."Product Name" = p.product_name
             AND s."Category"     = p.category
            GROUP BY s."Category"
            ORDER BY revenue_per_unit DESC
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Margin-proxy query failed: {result}"


# ---------------------------------------------------------------------------
# Three-table: sales1000 x sales_rep_targets x product_metrics
# ---------------------------------------------------------------------------

class TestThreeTableJoin:
    """Queries that touch all three tables simultaneously."""

    def test_rep_category_attainment(self):
        """Per rep+region+category: actual revenue, quota share, attainment."""
        sql = """
            SELECT
                s."Sales Rep"                                                  AS sales_rep,
                s."Region"                                                     AS region,
                s."Category",
                SUM(s."Total Revenue")                                          AS cat_revenue,
                t.quota_usd,
                ROUND(SUM(s."Total Revenue") / t.quota_usd, 4)                AS cat_attainment,
                p.unit_cost_usd                                                AS product_cost
            FROM sales1000 s
            JOIN sales_rep_targets t
              ON s."Sales Rep" = t.sales_rep
             AND s."Region"    = t.region
            JOIN product_metrics p
              ON s."Product Name" = p.product_name
             AND s."Category"     = p.category
            GROUP BY s."Sales Rep", s."Region", s."Category", t.quota_usd, p.unit_cost_usd
            ORDER BY cat_attainment DESC
            LIMIT 20
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Three-table query failed: {result}"
        parsed = json.loads(result)
        assert parsed["metadata"]["row_count"] >= 1

    def test_underperforming_reps_on_high_revenue_products(self):
        """Reps below quota who sell high-revenue products -- coaching targets."""
        sql = """
            WITH rep_actual AS (
                SELECT
                    s."Sales Rep" AS sales_rep,
                    s."Region"    AS region,
                    SUM(s."Total Revenue") AS actual
                FROM sales1000 s
                GROUP BY s."Sales Rep", s."Region"
            ),
            top_products AS (
                SELECT product_name, category
                FROM product_metrics
                ORDER BY unit_cost_usd DESC
                LIMIT 5
            )
            SELECT
                ra.sales_rep,
                ra.region,
                ra.actual,
                t.quota_usd,
                t.quota_usd - ra.actual AS gap
            FROM rep_actual ra
            JOIN sales_rep_targets t
              ON ra.sales_rep = t.sales_rep
             AND ra.region    = t.region
            WHERE ra.actual < t.quota_usd
              AND EXISTS (
                  SELECT 1
                  FROM sales1000 s2
                  JOIN top_products tp
                    ON s2."Product Name" = tp.product_name
                   AND s2."Category"     = tp.category
                  WHERE s2."Sales Rep" = ra.sales_rep
              )
            ORDER BY gap DESC
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Underperforming-reps query failed: {result}"


# ---------------------------------------------------------------------------
# Regression: single-table queries still work after multi-table wiring
# ---------------------------------------------------------------------------

class TestSingleTableRegression:
    """Ensure multi-table wiring doesn't break existing single-table paths."""

    def test_sales1000_aggregate(self):
        # "Total Revenue" is a Title Case column in the fact table
        sql = """
            SELECT "Region", SUM("Total Revenue") AS total
            FROM sales1000
            GROUP BY "Region"
            ORDER BY total DESC
        """
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Single-table query failed: {result}"
        parsed = json.loads(result)
        assert parsed["metadata"]["row_count"] >= 1

    def test_sales_rep_targets_standalone(self):
        # quota_usd is the column name in the independently generated dim table
        sql = "SELECT * FROM sales_rep_targets ORDER BY quota_usd DESC LIMIT 5"
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Targets standalone query failed: {result}"

    def test_product_metrics_standalone(self):
        # unit_cost_usd is the column name in product_metrics
        sql = "SELECT * FROM product_metrics ORDER BY unit_cost_usd DESC LIMIT 5"
        result = run_sql.invoke({"query": sql})
        assert not result.startswith("ERROR"), f"Product metrics standalone query failed: {result}"
