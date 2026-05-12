"""Generate sales_rep_targets.xlsx for multi-table integration testing.

This script derives the distinct (sales_rep, region) pairs from the already-loaded
sales1000 table in DuckDB, then assigns *independent* quota values so that
quota-vs-actual join queries produce meaningful, non-trivial results.

Usage:
    python scripts/generate_test_tables.py

Output:
    data/sales_rep_targets.xlsx

After running, load the file through the agent:
    load_file('data/sales_rep_targets.xlsx', 'sales_rep_targets')
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = os.getenv("DUCKDB_PATH", "./.local/duckdb/efficient-text-to-sql.duckdb")
OUT_PATH = Path("data/sales_rep_targets.xlsx")


def main() -> None:
    if not Path(DB_PATH).exists() and DB_PATH != ":memory:":
        raise SystemExit(
            f"DuckDB file not found at '{DB_PATH}'.\n"
            "Load sample_sales_1000.xlsx through the agent first."
        )

    conn = duckdb.connect(DB_PATH, read_only=True)

    try:
        pairs = conn.execute(
            "SELECT DISTINCT sales_rep, region FROM sales1000 ORDER BY sales_rep, region"
        ).df()
    except duckdb.CatalogException:
        raise SystemExit(
            "Table 'sales1000' not found in DuckDB.\n"
            "Load sample_sales_1000.xlsx through the agent first."
        )
    finally:
        conn.close()

    if pairs.empty:
        raise SystemExit("sales1000 exists but returned no rows.")

    # Independent quota values — NOT derived from historical revenue.
    # Using a fixed seed so the file is reproducible across runs.
    rng = np.random.default_rng(42)
    n = len(pairs)

    pairs["annual_quota"] = (
        rng.integers(80_000, 300_000, size=n).astype(float)
    )
    pairs["target_orders"] = rng.integers(20, 80, size=n)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_excel(OUT_PATH, index=False)

    print(f"Generated {n} rows  →  {OUT_PATH}")
    print("\nColumns:", list(pairs.columns))
    print("\nSample rows:")
    print(pairs.head(5).to_string(index=False))
    print(
        "\nNext step: load the file through the agent:\n"
        "  load_file('data/sales_rep_targets.xlsx', 'sales_rep_targets')"
    )


if __name__ == "__main__":
    main()
