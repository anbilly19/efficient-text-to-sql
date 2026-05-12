"""
scripts/seed_test_tables.py
----------------------------
Creates derived test tables in DuckDB from the already-loaded sales1000 fact table.
Run once after loading sample_sales_1000.xlsx via the agent loader.

Usage:
    python scripts/seed_test_tables.py [--db PATH]

The script is idempotent: it uses CREATE OR REPLACE TABLE so re-running is safe.
After creation it registers the tables in _data_registry so list_loaded_tables
and schema tools can see them.
"""

import argparse
import os
from pathlib import Path

import duckdb


FACT_TABLE = "sales1000"

SQL_REP_TARGETS = """
CREATE OR REPLACE TABLE sales_rep_targets AS
SELECT
    sales_rep,
    region,
    COUNT(*)                                AS orders_count,
    ROUND(SUM(total_revenue), 2)            AS historical_revenue,
    SUM(quantity)                           AS units_sold,
    ROUND(AVG(total_revenue), 2)            AS avg_order_value,
    ROUND(SUM(total_revenue) * 1.15, 2)    AS annual_quota
FROM {fact}
GROUP BY sales_rep, region
ORDER BY region, sales_rep;
""".format(fact=FACT_TABLE)

SQL_PRODUCT_METRICS = """
CREATE OR REPLACE TABLE product_metrics AS
SELECT
    product,
    category,
    COUNT(*)                             AS orders_count,
    SUM(quantity)                        AS total_units,
    ROUND(SUM(total_revenue), 2)         AS total_revenue,
    ROUND(AVG(unit_price), 2)            AS avg_unit_price,
    ROUND(AVG(total_revenue), 2)         AS avg_order_value
FROM {fact}
GROUP BY product, category
ORDER BY category, product;
""".format(fact=FACT_TABLE)

SQL_RELATIONSHIPS = """
CREATE TABLE IF NOT EXISTS _relationships (
    left_table   VARCHAR NOT NULL,
    left_col     VARCHAR NOT NULL,
    right_table  VARCHAR NOT NULL,
    right_col    VARCHAR NOT NULL,
    join_type    VARCHAR DEFAULT 'INNER',
    cardinality  VARCHAR DEFAULT 'N:1',
    description  VARCHAR,
    PRIMARY KEY (left_table, left_col, right_table, right_col)
);

INSERT OR IGNORE INTO _relationships VALUES
    ('sales1000', 'sales_rep', 'sales_rep_targets', 'sales_rep', 'LEFT', 'N:1',
     'Each sales1000 row maps to one rep-region row in sales_rep_targets'),
    ('sales1000', 'region',    'sales_rep_targets', 'region',    'LEFT', 'N:1',
     'Composite join partner: always join on BOTH sales_rep AND region'),
    ('sales1000', 'product',   'product_metrics',   'product',   'LEFT', 'N:1',
     'Each sales1000 row maps to one product row in product_metrics'),
    ('sales1000', 'category',  'product_metrics',   'category',  'LEFT', 'N:1',
     'Composite join partner: always join on BOTH product AND category');
"""

REGISTRY_UPSERT = """
INSERT OR REPLACE INTO _data_registry
    (dataset_name, parquet_path, source_file, row_count, column_count)
SELECT
    table_name,
    NULL,
    'derived:seed_test_tables.py',
    estimated_size,
    column_count
FROM (
    SELECT 'sales_rep_targets' AS table_name,
           (SELECT COUNT(*) FROM sales_rep_targets) AS estimated_size,
           7 AS column_count
    UNION ALL
    SELECT 'product_metrics',
           (SELECT COUNT(*) FROM product_metrics),
           7
)
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'sales_rep_targets'
);
"""


def seed(db_path: str) -> None:
    print(f"Connecting to: {db_path}")
    conn = duckdb.connect(db_path)

    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()}

    if FACT_TABLE not in tables:
        raise RuntimeError(
            f"Fact table '{FACT_TABLE}' not found. "
            "Load sample_sales_1000.xlsx via the agent first."
        )

    print("Creating sales_rep_targets ...")
    conn.execute(SQL_REP_TARGETS)
    rows_srt = conn.execute("SELECT COUNT(*) FROM sales_rep_targets").fetchone()[0]
    print(f"  -> {rows_srt} rows")

    print("Creating product_metrics ...")
    conn.execute(SQL_PRODUCT_METRICS)
    rows_pm = conn.execute("SELECT COUNT(*) FROM product_metrics").fetchone()[0]
    print(f"  -> {rows_pm} rows")

    print("Seeding _relationships ...")
    conn.execute(SQL_RELATIONSHIPS)
    rows_rel = conn.execute("SELECT COUNT(*) FROM _relationships").fetchone()[0]
    print(f"  -> {rows_rel} relationship rows")

    try:
        conn.execute(REGISTRY_UPSERT)
        print("  -> _data_registry updated")
    except Exception as e:
        print(f"  WARNING: Registry update skipped ({e})")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default=os.environ.get(
            "DUCKDB_PATH",
            str(Path(__file__).parent.parent / ".local" / "duckdb" / "efficient-text-to-sql.duckdb"),
        ),
        help="Path to DuckDB file (default: .local/duckdb/efficient-text-to-sql.duckdb)",
    )
    args = parser.parse_args()
    seed(args.db)
