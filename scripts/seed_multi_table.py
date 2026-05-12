#!/usr/bin/env python
"""
scripts/seed_multi_table.py

One-shot seeder that:
  1. Loads the three Excel tables into DuckDB via load_file.
  2. Registers every table in _data_registry.
  3. Indexes all columns in _column_catalog.
  4. Registers the authoritative join relationships.
  5. Upserts _table_context summaries.

Usage:
    # First generate the Excel files:
    python scripts/generate_test_data.py

    # Then seed (can be re-run safely -- all ops are idempotent):
    python scripts/seed_multi_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # Import here so DUCKDB_PATH env var is respected if set before running
    from agent.database import get_connection, register_relationship, index_table_schema
    from agent.db.catalog import upsert_table_context
    from agent.tools import load_file

    conn = get_connection()

    # ------------------------------------------------------------------
    # 1. Load source tables
    # ------------------------------------------------------------------
    tables = {
        "sales1000":         ROOT / "data" / "sample_sales_1000.xlsx",
        "sales_rep_targets": ROOT / "data" / "sales_rep_targets.xlsx",
        "product_metrics":   ROOT / "data" / "product_metrics.xlsx",
    }

    for dataset_name, path in tables.items():
        if not path.exists():
            print(f"SKIP {dataset_name}: {path} not found")
            continue
        result = load_file.invoke({"path": str(path), "dataset_name": dataset_name})
        print(result)

        # Ensure _data_registry entry exists with row/col counts
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{dataset_name}"').fetchone()[0]
        col_count = len(conn.execute(f'PRAGMA table_info("{dataset_name}")').fetchall())
        parquet_path = str(ROOT / "data" / "parquet" / f"{dataset_name}.parquet")
        conn.execute(
            """
            INSERT INTO _data_registry (dataset_name, parquet_path, row_count, column_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (dataset_name) DO UPDATE SET
                row_count    = excluded.row_count,
                column_count = excluded.column_count
            """,
            [dataset_name, parquet_path, row_count, col_count],
        )

        # Rebuild _column_catalog
        index_table_schema(conn, dataset_name)

    # ------------------------------------------------------------------
    # 2. Register authoritative join relationships
    #
    # sales1000 columns are Title Case  ("Sales Rep", "Region",
    #                                    "Product Name", "Category")
    # dim-table columns are snake_case  (sales_rep, region,
    #                                    product_name, category)
    # ------------------------------------------------------------------
    join_pairs = [
        # (left_table,      left_column,    right_table,         right_column,   cardinality,     description)
        ("sales1000",      "Sales Rep",    "sales_rep_targets", "sales_rep",    "many-to-one",   "Sales rep name join"),
        ("sales1000",      "Region",       "sales_rep_targets", "region",       "many-to-one",   "Region join"),
        ("sales1000",      "Product Name", "product_metrics",   "product_name", "many-to-one",   "Product name join"),
        ("sales1000",      "Category",     "product_metrics",   "category",     "many-to-one",   "Category join"),
    ]
    for left_tbl, left_col, right_tbl, right_col, card, desc in join_pairs:
        register_relationship(conn, left_tbl, left_col, right_tbl, right_col, card, desc)
        print(f"  registered: {left_tbl}.{left_col!r} -> {right_tbl}.{right_col!r}")

    # ------------------------------------------------------------------
    # 3. Upsert _table_context summaries
    # ------------------------------------------------------------------
    contexts = [
        (
            "sales1000",
            "Individual sales transactions. One row per order line.",
            "order_line",
            ["fact", "transactions", "sales"],
        ),
        (
            "sales_rep_targets",
            "Management quotas per (sales_rep, region) pair. One row per rep-region.",
            "sales_rep x region",
            ["dimension", "quota", "targets"],
        ),
        (
            "product_metrics",
            "Product catalogue with sourcing costs. One row per product.",
            "product",
            ["dimension", "product", "cost"],
        ),
    ]
    for dataset_name, summary, grain, tags in contexts:
        upsert_table_context(dataset_name, summary, grain, tags)
        print(f"  context upserted: {dataset_name}")

    print("\nSeed complete. Tables registered:")
    rows = conn.execute(
        "SELECT dataset_name, row_count, column_count FROM _data_registry ORDER BY dataset_name"
    ).fetchall()
    for name, rc, cc in rows:
        print(f"  {name}: {rc} rows, {cc} columns")

    print("\nRelationships registered:")
    rels = conn.execute(
        "SELECT left_table, left_column, right_table, right_column FROM _relationships ORDER BY relationship_id"
    ).fetchall()
    for lt, lc, rt, rc in rels:
        print(f"  {lt}.{lc!r} -> {rt}.{rc!r}")


if __name__ == "__main__":
    main()
