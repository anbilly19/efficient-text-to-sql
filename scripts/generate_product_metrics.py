"""
scripts/generate_product_metrics.py  —  Option A: independent product catalogue
--------------------------------------------------------------------------------
Builds data/product_metrics.xlsx as a STATIC CATALOGUE that is independent
of the sales figures in sales1000.

What this table is
------------------
product_metrics is a pre-existing product reference table — think of it as
the company's product master, filled with *catalogue* values and *prior-period*
baselines that existed before the orders in sample_sales_1000.xlsx were placed.

Because the values are NOT derived from sales1000, JOIN queries are genuinely
non-trivial: the agent has to compare actual transacted prices / revenues
against independent catalogue benchmarks.

How the catalogue is built
--------------------------
1. Read the unique (Product Name, Category) pairs from sample_sales_1000.xlsx
   so the join keys always match exactly.
2. Assign INDEPENDENT catalogue values using a per-product seeded RNG so the
   file is reproducible but not identical to anything the agent can compute:
     - list_price      : random draw per category price band
     - prior_revenue   : list_price * prior_units * discount factor
     - prior_units     : random draw, uncorrelated with sales1000 quantities
     - prior_orders    : prior_units / random items-per-order
     - avg_order_value : prior_revenue / prior_orders  (internally consistent)

Usage
-----
    python scripts/generate_product_metrics.py
    # reads  data/sample_sales_1000.xlsx
    # writes data/product_metrics.xlsx

    python scripts/generate_product_metrics.py \\
        --src path/to/sales.xlsx --out path/to/product_metrics.xlsx

Output columns
--------------
    product          VARCHAR  -- join key to sales1000
    category         VARCHAR  -- join key to sales1000
    list_price       DOUBLE   -- catalogue unit price (independent of sales)
    prior_revenue    DOUBLE   -- prior-period total revenue baseline
    prior_units      INTEGER  -- prior-period units sold
    prior_orders     INTEGER  -- prior-period order count
    avg_order_value  DOUBLE   -- prior_revenue / prior_orders
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SRC = Path("data/sample_sales_1000.xlsx")
DEFAULT_OUT = Path("data/product_metrics.xlsx")

COLUMNS = [
    "product",
    "category",
    "list_price",
    "prior_revenue",
    "prior_units",
    "prior_orders",
    "avg_order_value",
]

# Per-category price bands (min, max) USD — independent of actual sale prices.
# Extend this dict if new categories appear in the source data.
_PRICE_BANDS: dict[str, tuple[float, float]] = {
    "Electronics":     (49.99,  499.99),
    "Clothing":        (19.99,  149.99),
    "Home & Garden":   (24.99,  299.99),
    "Sports":          (14.99,  249.99),
    "Books":           ( 9.99,   49.99),
    "Toys":            ( 7.99,   89.99),
    "Food & Beverage": ( 3.99,   39.99),
    "Health":          (12.99,   99.99),
    "Automotive":      (29.99,  599.99),
    "Office":          (14.99,  199.99),
}
_DEFAULT_BAND = (9.99, 199.99)

# Column name normalisation (handles "Product Name", "product name", etc.)
_ALIASES = {
    "product name": "product",
    "product":      "product",
    "category":     "category",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _ALIASES[c.strip().lower()]
                               for c in df.columns
                               if c.strip().lower() in _ALIASES})


def _seed_for(product: str, category: str) -> int:
    """Deterministic per-product seed so reruns always produce the same file."""
    h = hashlib.md5(f"{product}|{category}".encode()).hexdigest()
    return int(h[:8], 16)


def _build_row(product: str, category: str) -> dict:
    rng = np.random.default_rng(_seed_for(product, category))

    lo, hi = _PRICE_BANDS.get(category, _DEFAULT_BAND)
    list_price = round(float(rng.uniform(lo, hi)), 2)

    # prior-period units: 200–2000, uncorrelated with sales1000
    prior_units = int(rng.integers(200, 2001))

    # items per order: 1–5
    items_per_order = float(rng.uniform(1.0, 5.0))
    prior_orders = max(1, round(prior_units / items_per_order))

    # prior revenue: list_price * units with a small volume-discount (0.85–1.0)
    discount = float(rng.uniform(0.85, 1.0))
    prior_revenue = round(list_price * prior_units * discount, 2)

    avg_order_value = round(prior_revenue / prior_orders, 2)

    return {
        "product":         product,
        "category":        category,
        "list_price":      list_price,
        "prior_revenue":   prior_revenue,
        "prior_units":     prior_units,
        "prior_orders":    prior_orders,
        "avg_order_value": avg_order_value,
    }


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
    df  = _normalise(raw)

    for col in ("product", "category"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found after normalisation.\n"
                f"Available: {list(df.columns)}"
            )

    pairs = (
        df[["product", "category"]]
        .drop_duplicates()
        .sort_values(["category", "product"])
        .reset_index(drop=True)
    )

    rows = [_build_row(r.product, r.category) for r in pairs.itertuples()]
    result = pd.DataFrame(rows, columns=COLUMNS)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(out, index=False)
    print(
        f"{len(result)} products across "
        f"{result['category'].nunique()} categories -> {out.resolve()}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data/product_metrics.xlsx — independent catalogue, not derived from sales figures"
    )
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="Source sales xlsx (join-key extraction only)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path (default: data/product_metrics.xlsx)")
    args = parser.parse_args()
    generate(args.src, args.out)
