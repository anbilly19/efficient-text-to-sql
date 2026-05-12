"""
tests/test_multi_table_seed.py
-------------------------------
Unit tests for scripts/seed_test_tables.py.

All tests run against an in-memory DuckDB fixture pre-loaded with a miniature
version of the sales1000 fact table -- no file I/O, no agent, no OpenAI key.

Run:
    pytest tests/test_multi_table_seed.py -v
"""

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import seed_test_tables as stt


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


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE sales1000 (
            order_id VARCHAR, order_date VARCHAR, sales_rep VARCHAR,
            region VARCHAR, product VARCHAR, category VARCHAR,
            quantity INTEGER, unit_price DOUBLE, total_revenue DOUBLE
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
    yield c
    c.close()


def seed_in_memory(c):
    c.execute(stt.SQL_REP_TARGETS)
    c.execute(stt.SQL_PRODUCT_METRICS)
    c.execute(stt.SQL_RELATIONSHIPS)
    try:
        c.execute(stt.REGISTRY_UPSERT)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

def test_sales_rep_targets_created(conn):
    seed_in_memory(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()}
    assert "sales_rep_targets" in tables


def test_product_metrics_created(conn):
    seed_in_memory(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()}
    assert "product_metrics" in tables


# ---------------------------------------------------------------------------
# Row counts
# ---------------------------------------------------------------------------

def test_sales_rep_targets_row_count(conn):
    seed_in_memory(conn)
    count = conn.execute("SELECT COUNT(*) FROM sales_rep_targets").fetchone()[0]
    assert count == 3


def test_product_metrics_row_count(conn):
    seed_in_memory(conn)
    count = conn.execute("SELECT COUNT(*) FROM product_metrics").fetchone()[0]
    assert count == 4


# ---------------------------------------------------------------------------
# sales_rep_targets correctness
# ---------------------------------------------------------------------------

def test_historical_revenue_alice(conn):
    seed_in_memory(conn)
    row = conn.execute("""
        SELECT historical_revenue FROM sales_rep_targets
        WHERE sales_rep = 'Alice Johnson' AND region = 'North'
    """).fetchone()
    assert row is not None
    assert abs(row[0] - 250.0) < 0.01


def test_orders_count_bob(conn):
    seed_in_memory(conn)
    row = conn.execute("""
        SELECT orders_count FROM sales_rep_targets
        WHERE sales_rep = 'Bob Smith' AND region = 'South'
    """).fetchone()
    assert row is not None
    assert row[0] == 2


def test_units_sold_carol(conn):
    seed_in_memory(conn)
    row = conn.execute("""
        SELECT units_sold FROM sales_rep_targets
        WHERE sales_rep = 'Carol White' AND region = 'East'
    """).fetchone()
    assert row is not None
    assert row[0] == 9


def test_annual_quota_is_115_pct(conn):
    seed_in_memory(conn)
    rows = conn.execute(
        "SELECT historical_revenue, annual_quota FROM sales_rep_targets"
    ).fetchall()
    assert len(rows) > 0
    for hist, quota in rows:
        assert abs(quota / hist - 1.15) < 0.001, f"Quota ratio wrong: {quota}/{hist}"


def test_avg_order_value_carol(conn):
    seed_in_memory(conn)
    row = conn.execute("""
        SELECT avg_order_value FROM sales_rep_targets
        WHERE sales_rep = 'Carol White' AND region = 'East'
    """).fetchone()
    assert row is not None
    assert abs(row[0] - 275.0) < 0.01


# ---------------------------------------------------------------------------
# product_metrics correctness
# ---------------------------------------------------------------------------

def test_product_metrics_keyboard_revenue(conn):
    seed_in_memory(conn)
    row = conn.execute("""
        SELECT total_revenue FROM product_metrics
        WHERE product = 'Keyboard' AND category = 'Electronics'
    """).fetchone()
    assert row is not None
    assert abs(row[0] - 600.0) < 0.01


def test_product_metrics_tshirt_units(conn):
    seed_in_memory(conn)
    row = conn.execute("""
        SELECT total_units FROM product_metrics
        WHERE product = 'T-Shirt' AND category = 'Clothing'
    """).fetchone()
    assert row is not None
    assert row[0] == 8


# ---------------------------------------------------------------------------
# _relationships registry
# ---------------------------------------------------------------------------

def test_relationships_row_count(conn):
    seed_in_memory(conn)
    count = conn.execute("SELECT COUNT(*) FROM _relationships").fetchone()[0]
    assert count == 4


def test_relationship_keys_present(conn):
    seed_in_memory(conn)
    rows = conn.execute("""
        SELECT left_table, left_col, right_table, right_col FROM _relationships
    """).fetchall()
    keys = {(r[0], r[1], r[2], r[3]) for r in rows}
    assert ("sales1000", "sales_rep", "sales_rep_targets", "sales_rep") in keys
    assert ("sales1000", "region",    "sales_rep_targets", "region")    in keys
    assert ("sales1000", "product",   "product_metrics",   "product")   in keys
    assert ("sales1000", "category",  "product_metrics",   "category")  in keys


def test_relationships_idempotent(conn):
    seed_in_memory(conn)
    conn.execute(stt.SQL_RELATIONSHIPS)
    count = conn.execute("SELECT COUNT(*) FROM _relationships").fetchone()[0]
    assert count == 4


# ---------------------------------------------------------------------------
# Fan-out guard
# ---------------------------------------------------------------------------

def test_composite_join_no_fan_out(conn):
    seed_in_memory(conn)
    fact_count = conn.execute("SELECT COUNT(*) FROM sales1000").fetchone()[0]
    joined_count = conn.execute("""
        SELECT COUNT(*)
        FROM sales1000 s
        LEFT JOIN sales_rep_targets t
          ON s.sales_rep = t.sales_rep AND s.region = t.region
    """).fetchone()[0]
    assert joined_count == fact_count, f"Fan-out: fact={fact_count}, joined={joined_count}"


def test_composite_join_returns_quota(conn):
    seed_in_memory(conn)
    rows = conn.execute("""
        SELECT s.order_id, t.annual_quota
        FROM sales1000 s
        LEFT JOIN sales_rep_targets t
          ON s.sales_rep = t.sales_rep AND s.region = t.region
    """).fetchall()
    for order_id, quota in rows:
        assert quota is not None and quota > 0, f"{order_id} has null/zero quota"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_full_reseed_idempotent(conn):
    seed_in_memory(conn)
    first = conn.execute("SELECT COUNT(*) FROM sales_rep_targets").fetchone()[0]
    seed_in_memory(conn)
    second = conn.execute("SELECT COUNT(*) FROM sales_rep_targets").fetchone()[0]
    assert first == second


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_fact_table_raises(tmp_path):
    db_path = str(tmp_path / "empty.duckdb")
    with duckdb.connect(db_path) as c:
        c.execute("CREATE TABLE dummy (x INT)")
    with pytest.raises(RuntimeError, match="sales1000"):
        stt.seed(db_path)
