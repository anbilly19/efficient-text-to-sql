#!/usr/bin/env python
"""
scripts/generate_test_tables.py
--------------------------------
Generates data/sales_rep_targets.xlsx from the actual sales1000 data.

Fixes vs the old version
------------------------
1. Produces ALL 48 (sales_rep x region) pairs that exist in sales1000,
   not a random subset.
2. Derives quota_usd from actual rep+region revenue so the values are
   realistic and ~30 % of reps fall *below* quota (interesting for queries).
3. Generates all 8 columns expected by seed_multi_table.py:
     sales_rep, region, quota_usd, fte_headcount, territory_tier,
     manager, hired_date, last_review_score

Usage
-----
    python scripts/generate_test_tables.py

    # Custom DB path:
    DUCKDB_PATH=/path/to/db.duckdb python scripts/generate_test_tables.py

Output
------
    data/sales_rep_targets.xlsx   (re-ingested by seed_multi_table.py)

After running:
    python scripts/seed_multi_table.py
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = os.getenv(
    "DUCKDB_PATH",
    str(ROOT / ".local" / "duckdb" / "efficient-text-to-sql.duckdb"),
)
OUT_PATH = ROOT / "data" / "sales_rep_targets.xlsx"

# Quota multipliers: ~30 % of rows get a multiplier < 1.0 (below quota).
# The rest are 1.05-1.30 (over quota but reachable).
_BELOW_QUOTA_SHARE = 0.30
_BELOW_RANGE = (0.70, 0.95)   # rep actual is 70-95 % of their quota
_ABOVE_RANGE = (1.05, 1.30)   # rep actual is 105-130 % of their quota

_TERRITORY_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
_MANAGERS = [
    "Sarah Connor", "John Doe", "Emily Clarke",
    "Marcus Reed",  "Linda Park",
]


def main() -> None:
    if not Path(DB_PATH).exists():
        raise SystemExit(
            f"DuckDB not found at '{DB_PATH}'.\n"
            "Load sample_sales_1000.xlsx through the agent first, then re-run."
        )

    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        # Actual revenue per (rep, region) — the ground truth we size quotas against.
        actuals = conn.execute(
            """
            SELECT sales_rep, region, SUM(total_revenue) AS actual_revenue
            FROM sales1000
            GROUP BY sales_rep, region
            ORDER BY sales_rep, region
            """
        ).df()
    except duckdb.CatalogException:
        raise SystemExit(
            "Table 'sales1000' not found.\n"
            "Load sample_sales_1000.xlsx through the agent first."
        )
    finally:
        conn.close()

    if actuals.empty:
        raise SystemExit("sales1000 returned no rows.")

    n   = len(actuals)
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ quotas
    # Assign ~30 % of reps a multiplier that puts them *below* quota.
    n_below  = max(1, round(n * _BELOW_QUOTA_SHARE))
    below_ix = rng.choice(n, size=n_below, replace=False)
    below_mask = np.zeros(n, dtype=bool)
    below_mask[below_ix] = True

    multipliers = np.where(
        below_mask,
        # quota > actual  →  rep is below quota
        actuals["actual_revenue"] / rng.uniform(*_BELOW_RANGE, size=n),
        # quota < actual  →  rep is above quota
        actuals["actual_revenue"] / rng.uniform(*_ABOVE_RANGE, size=n),
    )
    actuals["quota_usd"] = multipliers.round(0).astype(int)

    # ------------------------------------------------------------------ extras
    actuals["fte_headcount"]     = rng.integers(1, 6, size=n)
    actuals["territory_tier"]    = rng.choice(_TERRITORY_TIERS, size=n)
    actuals["manager"]           = rng.choice(_MANAGERS, size=n)

    base_date = date(2018, 1, 1)
    actuals["hired_date"] = [
        base_date + timedelta(days=int(d))
        for d in rng.integers(0, 365 * 6, size=n)
    ]
    actuals["last_review_score"] = rng.uniform(2.5, 5.0, size=n).round(2)

    # ------------------------------------------------------------------ output
    out_cols = [
        "sales_rep", "region", "quota_usd",
        "fte_headcount", "territory_tier", "manager",
        "hired_date", "last_review_score",
    ]
    df = actuals[out_cols].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_PATH, index=False)

    n_below_actual = below_mask.sum()
    print(f"Generated {n} rows  ->  {OUT_PATH}")
    print(f"  Below quota : {n_below_actual} reps ({n_below_actual/n*100:.0f}%)")
    print(f"  Above quota : {n - n_below_actual} reps")
    print("\nSample rows:")
    print(df.head(8).to_string(index=False))
    print("\nNext: run  python scripts/seed_multi_table.py  to reload into DuckDB.")


if __name__ == "__main__":
    main()
