"""
scripts/generate_product_metrics.py
-------------------------------------
Derives data/product_metrics.xlsx directly from data/sample_sales_1000.xlsx
using pandas groupby -- no DuckDB, no agent, no hardcoded rows.

The output mirrors the product_metrics table the agent uses so that
round_12 JOIN queries have real, consistent baselines.

Usage:
    python scripts/generate_product_metrics.py
    # reads  data/sample_sales_1000.xlsx
    # writes data/product_metrics.xlsx

    python scripts/generate_product_metrics.py \
        --src path/to/sales.xlsx \
        --out path/to/product_metrics.xlsx

Output columns:
    product         -- join key (maps to sales1000 "Product Name")
    category        -- join key (maps to sales1000 "Category")
    total_revenue   -- SUM(Total Revenue)
    total_units     -- SUM(Quantity)
    order_count     -- COUNT(*)
    avg_unit_price  -- MEAN(Unit Price)
    avg_order_value -- total_revenue / order_count
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SRC = Path("data/sample_sales_1000.xlsx")
DEFAULT_OUT = Path("data/product_metrics.xlsx")

COLUMNS = [
    "product",
    "category",
    "total_revenue",
    "total_units",
    "order_count",
    "avg_unit_price",
    "avg_order_value",
]

# Flexible column name mapping: keys are what we look for (lowercase, stripped),
# values are the canonical internal names used in the groupby.
_COL_ALIASES: dict[str, str] = {
    "product name": "product",
    "product":      "product",
    "category":     "category",
    "total revenue": "total_revenue",
    "revenue":       "total_revenue",
    "quantity":      "quantity",
    "qty":           "quantity",
    "unit price":    "unit_price",
    "price":         "unit_price",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw xlsx columns to internal canonical names."""
    rename = {}
    for raw in df.columns:
        alias = _COL_ALIASES.get(raw.strip().lower())
        if alias:
            rename[raw] = alias
    return df.rename(columns=rename)


def generate(
    src: str | Path = DEFAULT_SRC,
    out: str | Path = DEFAULT_OUT,
) -> pd.DataFrame:
    src, out = Path(src), Path(out)
    if not src.exists():
        raise FileNotFoundError(
            f"Source file not found: {src.resolve()}\n"
            "Pass --src to specify a different path."
        )

    raw = pd.read_excel(src)
    df  = _normalise_columns(raw)

    missing = {c for c in ("product", "category", "total_revenue", "quantity", "unit_price")
               if c not in df.columns}
    if missing:
        raise ValueError(
            f"Required columns not found after normalisation: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    agg = (
        df.groupby(["product", "category"], sort=True)
        .agg(
            total_revenue  =("total_revenue", "sum"),
            total_units    =("quantity",       "sum"),
            order_count    =("total_revenue",  "count"),
            avg_unit_price =("unit_price",     "mean"),
        )
        .reset_index()
    )

    agg["avg_order_value"] = (agg["total_revenue"] / agg["order_count"]).round(2)
    agg["total_revenue"]   = agg["total_revenue"].round(2)
    agg["avg_unit_price"]  = agg["avg_unit_price"].round(2)
    agg["total_units"]     = agg["total_units"].astype(int)
    agg["order_count"]     = agg["order_count"].astype(int)

    result = agg[COLUMNS]

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(out, index=False)
    print(f"Derived {len(result)} product rows from {src.name} -> {out.resolve()}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Derive data/product_metrics.xlsx from sample_sales_1000.xlsx"
    )
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="Source sales xlsx")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path")
    args = parser.parse_args()
    generate(args.src, args.out)
