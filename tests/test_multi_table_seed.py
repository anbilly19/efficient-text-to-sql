"""
tests/test_multi_table_seed.py
-------------------------------
Unit tests for scripts/seed_multi_table.py.

All tests run against an in-memory DuckDB fixture — no file I/O, no agent, no API key.

Run the whole file:
    pytest tests/test_multi_table_seed.py -v

Run a single class:
    pytest tests/test_multi_table_seed.py::TestRepTargets -v

Run a single test:
    pytest tests/test_multi_table_seed.py::TestRepTargets::test_annual_quota_is_115_pct -v

Run with stdout:
    pytest tests/test_multi_table_seed.py -v -s
"""

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import seed_multi_table as smt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINI_SALES = [
    ("ORD-001", "2024-01-10", "Alice Johnson", "North", "Keyboard", "Electronics", 2, 100.00, 200.00),
    ("ORD-002", "2024-01-11", "Alice Johnson", "North", "Mouse",    "Electronics", 1,  50.00,  50.00),
    ("ORD-003", "2024-01-12", "Bob Smith",     "South", "T-Shirt",  "Clothing",    3,  30.00,  90.00),
    ("ORD-004", "2024-01-13", "Bob Smith",     "South", "Jeans",    "Clothing",    2,  60.00, 120.00),
    ("ORD-005", "2024-01-14", "Carol White",   "East",  "Keyboard", "Electronics", 4, 100.00, 400.00),
    ("ORD-006", "2024-01-15", "Carol White",   "East",  "T-Shirt",  "Clothing",    5,  30.00, 150.00),
]

COLUMNS = (
    "order_id", "order_date", "Sales Rep", "Region",
    "Product Name", "Category", "Quantity Ordered", "Unit Price", "Total Revenue"
)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    cols = ", ".join(f'"{col}" VARCHAR' if col not in ("Quantity Ordered",) else f'"{col}" INTEGER'
                     for col in COLUMNS)
    # Use the exact column names from sample_sales_1000.xlsx
    c.execute("""
        CREATE TABLE sales1000 (
            order_id        VARCHAR,
            order_date      VARCHAR,
            "Sales Rep"     VARCHAR,
            "Region"        VARCHAR,
            "Product Name"  VARCHAR,
            "Category"      VARCHAR,
            "Quantity Ordered" INTEGER,
            "Unit Price"    DOUBLE,
            "Total Revenue" DOUBLE
        )
    """)
    c.executemany("INSERT INTO sales1000 VALUES (?,?,?,?,?,?,?,?,?)", MINI_SALES)
    c.execute("""
        CREATE TABLE IF NOT EXISTS _data_registry (
            dataset_name  VARCHAR NOT NULL PRIMARY KEY,
            parquet_path  VARCHAR,
            source_file   VARCHAR,
            row_count     BIGINT,
            column_count  INTEGER,
            ingested_at   TIMESTAMP DEFAULT current_timestamp
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS _relationships (
            id          INTEGER PRIMARY KEY,
            left_table  VARCHAR,
            left_col    VARCHAR,
            right_table VARCHAR,
            right_col   VARCHAR,
            join_type   VARCHAR DEFAULT 'many-to-one'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS _table_context (
            table_name  VARCHAR PRIMARY KEY,
            summary     VARCHAR,
            tags        VARCHAR
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS _column_catalog (
            table_name   VARCHAR,
            column_name  VARCHAR,
            is_metric    BOOLEAN DEFAULT FALSE,
            is_dimension BOOLEAN DEFAULT FALSE,
            is_join_key  BOOLEAN DEFAULT FALSE
        )
    """)
    yield c
    c.close()


def _seed(c):
    """Run all seed SQL statements against the given connection."""
    c.execute(smt.SQL_REP_TARGETS)
    c.execute(smt.SQL_PRODUCT_METRICS)
    c.execute(smt.SQL_RELATIONSHIPS)
    try:
        c.execute(smt.REGISTRY_UPSERT)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

class TestTableCreation:
    """Run: pytest tests/test_multi_table_seed.py::TestTableCreation -v"""

    def test_sales_rep_targets_created(self, conn):
        _seed(conn)
        tables = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        assert "sales_rep_targets" in tables

    def test_product_metrics_created(self, conn):
        _seed(conn)
        tables = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        assert "product_metrics" in tables


# ---------------------------------------------------------------------------
# Row counts
# ---------------------------------------------------------------------------

class TestRowCounts:
    """Run: pytest tests/test_multi_table_seed.py::TestRowCounts -v"""

    def test_sales_rep_targets_row_count(self, conn):
        _seed(conn)
        count = conn.execute("SELECT COUNT(*) FROM sales_rep_targets").fetchone()[0]
        assert count == 3  # Alice/North, Bob/South, Carol/East

    def test_product_metrics_row_count(self, conn):
        _seed(conn)
        count = conn.execute("SELECT COUNT(*) FROM product_metrics").fetchone()[0]
        assert count == 4  # Keyboard/Electronics, Mouse/Electronics, T-Shirt/Clothing, Jeans/Clothing


# ---------------------------------------------------------------------------
# sales_rep_targets correctness
# ---------------------------------------------------------------------------

class TestRepTargets:
    """Run: pytest tests/test_multi_table_seed.py::TestRepTargets -v"""

    def test_historical_revenue_alice(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT historical_revenue FROM sales_rep_targets
            WHERE sales_rep = 'Alice Johnson' AND region = 'North'
        """).fetchone()
        assert row is not None
        assert abs(row[0] - 250.0) < 0.01

    def test_orders_count_bob(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT orders_count FROM sales_rep_targets
            WHERE sales_rep = 'Bob Smith' AND region = 'South'
        """).fetchone()
        assert row is not None
        assert row[0] == 2

    def test_units_sold_carol(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT units_sold FROM sales_rep_targets
            WHERE sales_rep = 'Carol White' AND region = 'East'
        """).fetchone()
        assert row is not None
        assert row[0] == 9

    def test_annual_quota_is_115_pct(self, conn):
        _seed(conn)
        rows = conn.execute(
            "SELECT historical_revenue, annual_quota FROM sales_rep_targets"
        ).fetchall()
        assert len(rows) > 0
        for hist, quota in rows:
            assert abs(quota / hist - 1.15) < 0.001, f"Quota ratio wrong: {quota}/{hist}"

    def test_avg_order_value_carol(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT avg_order_value FROM sales_rep_targets
            WHERE sales_rep = 'Carol White' AND region = 'East'
        """).fetchone()
        assert row is not None
        assert abs(row[0] - 275.0) < 0.01


# ---------------------------------------------------------------------------
# product_metrics correctness
# ---------------------------------------------------------------------------

class TestProductMetrics:
    """Run: pytest tests/test_multi_table_seed.py::TestProductMetrics -v"""

    def test_keyboard_revenue(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT total_revenue FROM product_metrics
            WHERE product = 'Keyboard' AND category = 'Electronics'
        """).fetchone()
        assert row is not None
        assert abs(row[0] - 600.0) < 0.01  # ORD-001 (200) + ORD-005 (400)

    def test_tshirt_units(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT total_units FROM product_metrics
            WHERE product = 'T-Shirt' AND category = 'Clothing'
        """).fetchone()
        assert row is not None
        assert row[0] == 8  # ORD-003 (3) + ORD-006 (5)

    def test_avg_unit_price_mouse(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT avg_unit_price FROM product_metrics
            WHERE product = 'Mouse' AND category = 'Electronics'
        """).fetchone()
        assert row is not None
        assert abs(row[0] - 50.0) < 0.01

    def test_order_count_jeans(self, conn):
        _seed(conn)
        row = conn.execute("""
            SELECT order_count FROM product_metrics
            WHERE product = 'Jeans' AND category = 'Clothing'
        """).fetchone()
        assert row is not None
        assert row[0] == 1


# ---------------------------------------------------------------------------
# _relationships registry
# ---------------------------------------------------------------------------

class TestRelationshipsRegistry:
    """Run: pytest tests/test_multi_table_seed.py::TestRelationshipsRegistry -v"""

    def test_row_count(self, conn):
        _seed(conn)
        count = conn.execute("SELECT COUNT(*) FROM _relationships").fetchone()[0]
        assert count == 4

    def test_expected_keys_present(self, conn):
        _seed(conn)
        rows = conn.execute(
            "SELECT left_table, left_col, right_table, right_col FROM _relationships"
        ).fetchall()
        keys = {(r[0], r[1], r[2], r[3]) for r in rows}
        assert ("sales1000", "Sales Rep",    "sales_rep_targets", "sales_rep") in keys
        assert ("sales1000", "Region",        "sales_rep_targets", "region")    in keys
        assert ("sales1000", "Product Name",  "product_metrics",   "product")   in keys
        assert ("sales1000", "Category",      "product_metrics",   "category")  in keys

    def test_idempotent(self, conn):
        _seed(conn)
        conn.execute(smt.SQL_RELATIONSHIPS)
        count = conn.execute("SELECT COUNT(*) FROM _relationships").fetchone()[0]
        assert count == 4


# ---------------------------------------------------------------------------
# Fan-out guards
# ---------------------------------------------------------------------------

class TestFanOut:
    """Run: pytest tests/test_multi_table_seed.py::TestFanOut -v"""

    def test_rep_targets_no_fanout(self, conn):
        _seed(conn)
        fact_count = conn.execute("SELECT COUNT(*) FROM sales1000").fetchone()[0]
        joined = conn.execute("""
            SELECT COUNT(*)
            FROM sales1000 s
            LEFT JOIN sales_rep_targets t
              ON s."Sales Rep" = t.sales_rep AND s."Region" = t.region
        """).fetchone()[0]
        assert joined == fact_count, f"Fan-out: fact={fact_count}, joined={joined}"

    def test_product_metrics_no_fanout(self, conn):
        _seed(conn)
        fact_count = conn.execute("SELECT COUNT(*) FROM sales1000").fetchone()[0]
        joined = conn.execute("""
            SELECT COUNT(*)
            FROM sales1000 s
            LEFT JOIN product_metrics p
              ON s."Product Name" = p.product AND s."Category" = p.category
        """).fetchone()[0]
        assert joined == fact_count, f"Fan-out: fact={fact_count}, joined={joined}"

    def test_quota_lookup_returns_value(self, conn):
        _seed(conn)
        rows = conn.execute("""
            SELECT s.order_id, t.annual_quota
            FROM sales1000 s
            LEFT JOIN sales_rep_targets t
              ON s."Sales Rep" = t.sales_rep AND s."Region" = t.region
        """).fetchall()
        for order_id, quota in rows:
            assert quota is not None and quota > 0, f"{order_id} has null/zero quota"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Run: pytest tests/test_multi_table_seed.py::TestIdempotency -v"""

    def test_full_reseed_idempotent(self, conn):
        _seed(conn)
        first = conn.execute("SELECT COUNT(*) FROM sales_rep_targets").fetchone()[0]
        _seed(conn)
        second = conn.execute("SELECT COUNT(*) FROM sales_rep_targets").fetchone()[0]
        assert first == second

    def test_product_metrics_reseed_idempotent(self, conn):
        _seed(conn)
        first = conn.execute("SELECT COUNT(*) FROM product_metrics").fetchone()[0]
        _seed(conn)
        second = conn.execute("SELECT COUNT(*) FROM product_metrics").fetchone()[0]
        assert first == second


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Run: pytest tests/test_multi_table_seed.py::TestErrorHandling -v"""

    def test_missing_fact_table_raises(self, tmp_path):
        db_path = str(tmp_path / "empty.duckdb")
        with duckdb.connect(db_path) as c:
            c.execute("CREATE TABLE dummy (x INT)")
        with pytest.raises(RuntimeError, match="sales1000"):
            smt.seed(db_path)
