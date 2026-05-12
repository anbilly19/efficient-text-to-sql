#!/usr/bin/env python
"""
scripts/seed_multi_table.py

Materialises sales_rep_targets and product_metrics from sales1000,
registers their authoritative join relationships, and seeds _table_context.

Usage:
    # From the project root:
    python scripts/seed_multi_table.py

    # Or with a custom DB path:
    DUCKDB_PATH=/path/to/db.duckdb python scripts/seed_multi_table.py

Pre-requisite: sales1000 must already be loaded (load file at ... as sales1000).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `agent` is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.database import get_connection, index_table_schema, register_relationship  # noqa: E402
from agent.db.catalog import upsert_table_context  # noqa: E402


def _run_seed_sql(conn, sql_path: Path) -> None:
    sql = sql_path.read_text()
    # Split on semicolons and run each statement individually (DuckDB limitation)
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            conn.execute(stmt)
    print(f"  ✓  Executed {sql_path.name}")


def main() -> None:
    conn = get_connection()

    # ── 0. Verify sales1000 exists ──────────────────────────────────────────
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'sales1000'"
    ).fetchone()
    if not row:
        print(
            "ERROR: sales1000 table not found. "
            "Load the source file first:\n"
            "  load file at data/sample_sales_1000.xlsx as sales1000"
        )
        sys.exit(1)

    seeds_dir = PROJECT_ROOT / "agent" / "data" / "seeds"

    # ── 1. Materialise derived tables ─────────────────────────────────────
    print("Creating derived tables...")
    _run_seed_sql(conn, seeds_dir / "sales_rep_targets.sql")
    _run_seed_sql(conn, seeds_dir / "product_metrics.sql")

    # ── 2. Register into _data_registry (so the agent can see them) ───────
    print("Registering tables in _data_registry...")
    for table_name in ("sales_rep_targets", "product_metrics"):
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        col_count = conn.execute(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO _data_registry (dataset_name, parquet_path, source_file, row_count, column_count)
            VALUES (?, '', 'derived', ?, ?)
            ON CONFLICT (dataset_name) DO UPDATE SET
                row_count    = excluded.row_count,
                column_count = excluded.column_count,
                ingested_at  = current_timestamp
            """,
            [table_name, row_count, col_count],
        )
        print(f"  ✓  Registered {table_name} ({row_count} rows, {col_count} cols)")

    # ── 3. Index column catalog ────────────────────────────────────────────
    print("Indexing column catalog...")
    for table_name in ("sales_rep_targets", "product_metrics"):
        index_table_schema(conn, table_name)
        print(f"  ✓  Indexed {table_name}")

    # ── 4. Register authoritative relationships ────────────────────────────
    print("Registering join relationships...")
    rels = [
        (
            "sales1000", "Sales Rep",
            "sales_rep_targets", "sales_rep",
            "many-to-one",
            "Each sales1000 row belongs to one sales rep target row",
        ),
        (
            "sales1000", "Region",
            "sales_rep_targets", "region",
            "many-to-one",
            "Each sales1000 row belongs to one region in sales_rep_targets",
        ),
        (
            "sales1000", "Product Name",
            "product_metrics", "product_name",
            "many-to-one",
            "Each sales1000 row joins to one product_metrics row",
        ),
    ]
    for lt, lc, rt, rc, card, desc in rels:
        register_relationship(conn, lt, lc, rt, rc, card, desc)
        print(f"  ✓  {lt}.{lc!r} → {rt}.{rc!r} ({card})")

    # ── 5. Upsert _table_context ───────────────────────────────────────────
    print("Setting table context...")
    upsert_table_context(
        "sales_rep_targets",
        summary="Per-sales-rep historical revenue and quota targets, derived from sales1000.",
        grain="one row per sales_rep + region",
        tags=["targets", "quota", "sales_rep", "region"],
        conn=conn,
    )
    upsert_table_context(
        "product_metrics",
        summary="Per-product aggregated sales volume and revenue, derived from sales1000.",
        grain="one row per product_name",
        tags=["product", "volume", "revenue", "category"],
        conn=conn,
    )
    print("  ✓  Table context updated")

    print("\n✅ Multi-table seed complete.")
    print("   Tables available: sales1000, sales_rep_targets, product_metrics")


if __name__ == "__main__":
    main()
