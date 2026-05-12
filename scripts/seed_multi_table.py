#!/usr/bin/env python
"""
scripts/seed_multi_table.py

One-shot seeder that:
  1. Forces PARQUET_STORE to the canonical .local/parquet directory so the
     parquet files are ALWAYS written there, regardless of any PARQUET_STORE
     env var that may be set by pytest or a shell session.
  2. Loads the three Excel tables into DuckDB via load_file.
  3. Purges ALL inferred relationships for the three tables.
  4. Registers the four authoritative join pairs.
  5. Upserts _table_context summaries.

Usage:
    python scripts/generate_test_data.py   # only once
    python scripts/seed_multi_table.py     # idempotent, re-run any time
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ["sales1000", "sales_rep_targets", "product_metrics"]

# ---------------------------------------------------------------------------
# CRITICAL: force PARQUET_STORE to the canonical location BEFORE any agent
# module is imported, so load_file always writes to .local/parquet/ and the
# stored parquet_path in _data_registry is stable across test runs.
# ---------------------------------------------------------------------------
_CANONICAL_PARQUET_STORE = ROOT / ".local" / "parquet"
os.environ["PARQUET_STORE"] = str(_CANONICAL_PARQUET_STORE)

# Purge any already-imported agent modules so they re-bind to the new env var.
for _mod in [m for m in sys.modules if m.startswith("agent")]:
    del sys.modules[_mod]


def main() -> None:
    from agent.database import get_connection, register_relationship, index_table_schema
    from agent.db.catalog import upsert_table_context
    from agent.tools import load_file

    conn = get_connection()

    # ------------------------------------------------------------------
    # 1. Load source tables
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Purge inferred relationships (auto-detected junk) for these tables
    # ------------------------------------------------------------------
    print("\nPurging inferred relationships for seeded tables...")
    conn.execute(
        """
        DELETE FROM _relationships
        WHERE (left_table = ANY(?) OR right_table = ANY(?))
          AND cardinality = 'inferred'
        """,
        [TABLES, TABLES],
    )

    # ------------------------------------------------------------------
    # 3. Register the four authoritative join pairs
    # ------------------------------------------------------------------
    join_pairs = [
        ("sales1000", "Sales Rep",    "sales_rep_targets", "sales_rep",    "many-to-one", "Sales rep name join"),
        ("sales1000", "Region",       "sales_rep_targets", "region",       "many-to-one", "Region join"),
        ("sales1000", "Product Name", "product_metrics",   "product_name", "many-to-one", "Product name join"),
        ("sales1000", "Category",     "product_metrics",   "category",     "many-to-one", "Category join"),
    ]
    for left_tbl, left_col, right_tbl, right_col, card, desc in join_pairs:
        register_relationship(conn, left_tbl, left_col, right_tbl, right_col, card, desc)
        print(f"  registered: {left_tbl}.{left_col!r} -> {right_tbl}.{right_col!r}")

    # ------------------------------------------------------------------
    # 4. Upsert _table_context summaries
    # ------------------------------------------------------------------
    contexts = [
        ("sales1000",         "Individual sales transactions. One row per order line.",             "order_line",        ["fact", "transactions", "sales"]),
        ("sales_rep_targets", "Management quotas per (sales_rep, region) pair.",                   "sales_rep x region", ["dimension", "quota", "targets"]),
        ("product_metrics",   "Product catalogue with sourcing costs. One row per product.",        "product",           ["dimension", "product", "cost"]),
    ]
    for dataset_name, summary, grain, tags in contexts:
        upsert_table_context(dataset_name, summary, grain, tags)
        print(f"  context upserted: {dataset_name}")

    print("\nSeed complete. Tables registered:")
    for name, rc, cc in conn.execute(
        "SELECT dataset_name, row_count, column_count FROM _data_registry ORDER BY dataset_name"
    ).fetchall():
        print(f"  {name}: {rc} rows, {cc} columns")

    print("\nRelationships registered:")
    for lt, lc, rt, rc, card in conn.execute(
        "SELECT left_table, left_column, right_table, right_column, cardinality "
        "FROM _relationships ORDER BY relationship_id"
    ).fetchall():
        print(f"  {lt}.{lc!r} -> {rt}.{rc!r}  ({card})")

    print(f"\nParquets written to: {_CANONICAL_PARQUET_STORE}")


if __name__ == "__main__":
    main()
