#!/usr/bin/env python
"""
scripts/seed_multi_table.py  (idempotent)

Column reference
----------------
sales1000:         order_id, order_date, sales_rep, region, product,
                   category, quantity, unit_price, total_revenue
sales_rep_targets: sales_rep, region, quota_usd, fte_headcount,
                   territory_tier, manager, hired_date, last_review_score
product_metrics:   product_name, category, unit_cost_usd, launch_year,
                   lifecycle_stage, supplier, sku, reorder_point_units

Join keys
---------
  sales1000.sales_rep  -> sales_rep_targets.sales_rep
  sales1000.region     -> sales_rep_targets.region
  sales1000.product    -> product_metrics.product_name   (different names!)
  sales1000.category   -> product_metrics.category
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ["sales1000", "sales_rep_targets", "product_metrics"]

_CANONICAL_PARQUET_STORE = ROOT / ".local" / "parquet"
os.environ["PARQUET_STORE"] = str(_CANONICAL_PARQUET_STORE)

for _mod in [m for m in sys.modules if m.startswith("agent")]:
    del sys.modules[_mod]


def main() -> None:
    from agent.database import get_connection, register_relationship, index_table_schema
    from agent.db.catalog import upsert_table_context
    from agent.tools import load_file

    conn = get_connection()

    table_paths = {
        "sales1000":         ROOT / "data" / "sample_sales_1000.xlsx",
        "sales_rep_targets": ROOT / "data" / "sales_rep_targets.xlsx",
        "product_metrics":   ROOT / "data" / "product_metrics.xlsx",
    }

    for dataset_name, path in table_paths.items():
        if not path.exists():
            print(f"SKIP {dataset_name}: {path} not found")
            continue
        result = load_file.invoke({"path": str(path), "dataset_name": dataset_name})
        print(result)

        row_count = conn.execute(f'SELECT COUNT(*) FROM "{dataset_name}"').fetchone()[0]
        col_count = len(conn.execute(f'PRAGMA table_info("{dataset_name}")').fetchall())
        parquet_path = str(_CANONICAL_PARQUET_STORE / f"{dataset_name}.parquet")
        conn.execute(
            """
            INSERT INTO _data_registry (dataset_name, parquet_path, row_count, column_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (dataset_name) DO UPDATE SET
                parquet_path = excluded.parquet_path,
                row_count    = excluded.row_count,
                column_count = excluded.column_count
            """,
            [dataset_name, parquet_path, row_count, col_count],
        )
        index_table_schema(conn, dataset_name)

    print("\nPurging inferred relationships for seeded tables...")
    conn.execute(
        """
        DELETE FROM _relationships
        WHERE (left_table = ANY(?) OR right_table = ANY(?))
          AND cardinality = 'inferred'
        """,
        [TABLES, TABLES],
    )

    # sales1000.product -> product_metrics.product_name  (different column names)
    join_pairs = [
        ("sales1000", "sales_rep", "sales_rep_targets", "sales_rep",   "many-to-one", "Sales rep join"),
        ("sales1000", "region",    "sales_rep_targets", "region",       "many-to-one", "Region join"),
        ("sales1000", "product",   "product_metrics",   "product_name", "many-to-one", "Product join"),
        ("sales1000", "category",  "product_metrics",   "category",     "many-to-one", "Category join"),
    ]
    for left_tbl, left_col, right_tbl, right_col, card, desc in join_pairs:
        register_relationship(conn, left_tbl, left_col, right_tbl, right_col, card, desc)
        print(f"  registered: {left_tbl}.{left_col} -> {right_tbl}.{right_col}")

    contexts = [
        ("sales1000",         "Individual sales transactions. One row per order line.",      "order_line",         ["fact", "transactions", "sales"]),
        ("sales_rep_targets", "Management quotas per (sales_rep, region) pair.",             "sales_rep x region", ["dimension", "quota", "targets"]),
        ("product_metrics",   "Product catalogue with sourcing costs. One row per product.", "product",            ["dimension", "product", "cost"]),
    ]
    for dataset_name, summary, grain, tags in contexts:
        upsert_table_context(dataset_name, summary, grain, tags)
        print(f"  context upserted: {dataset_name}")

    print("\nSeed complete.")
    for name, rc, cc in conn.execute(
        "SELECT dataset_name, row_count, column_count FROM _data_registry ORDER BY dataset_name"
    ).fetchall():
        print(f"  {name}: {rc} rows, {cc} cols")

    print("\nRelationships:")
    for lt, lc, rt, rc_col, card in conn.execute(
        "SELECT left_table, left_column, right_table, right_column, cardinality "
        "FROM _relationships ORDER BY relationship_id"
    ).fetchall():
        print(f"  {lt}.{lc} -> {rt}.{rc_col}  ({card})")

    print(f"\nParquets at: {_CANONICAL_PARQUET_STORE}")


if __name__ == "__main__":
    main()
