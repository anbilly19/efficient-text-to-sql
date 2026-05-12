#!/usr/bin/env python
"""
scripts/generate_test_data.py

Generates two independent Excel files that can be uploaded to the agent
to test multi-table JOIN queries:

  data/sales_rep_targets.xlsx
      Quota and headcount data for each (sales_rep, region) pair.
      Join key to sales1000:  "Sales Rep" <-> sales_rep
                              "Region"    <-> region
      Columns (all independently authored — NOT derived from sales1000):
        sales_rep, region, quota_usd, fte_headcount, territory_tier,
        manager, hired_date, last_review_score

  data/product_metrics.xlsx
      Catalogue / product-management data for each product.
      Join key to sales1000:  "Product Name" <-> product_name
      Columns:
        product_name, category, unit_cost_usd, launch_year,
        lifecycle_stage, supplier, sku, reorder_point_units

Usage:
    python scripts/generate_test_data.py          # writes to data/
    python scripts/generate_test_data.py --out /tmp/test_data

No external dependencies beyond openpyxl / pandas (already in the venv).
"""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# These values MUST match the distinct values in sample_sales_1000.xlsx so
# that JOIN queries return non-empty results when the agent is tested.
# They are used ONLY as key columns — all other columns are independently made.
# ---------------------------------------------------------------------------

SALES_REPS = [
    "Alice Morgan", "Bob Nguyen", "Carol Smith", "David Lee",
    "Eva Fischer", "Frank Torres", "Grace Kim", "Henry Patel",
    "Irene Dubois", "Jake Wilson",
]

REGIONS = ["North", "South", "East", "West", "Central"]

# (sales_rep, region) pairs — each rep covers 1–2 regions
REP_REGION_PAIRS: list[tuple[str, str]] = [
    ("Alice Morgan",  "North"),
    ("Alice Morgan",  "East"),
    ("Bob Nguyen",    "South"),
    ("Carol Smith",   "West"),
    ("Carol Smith",   "Central"),
    ("David Lee",     "North"),
    ("Eva Fischer",   "East"),
    ("Frank Torres",  "South"),
    ("Frank Torres",  "West"),
    ("Grace Kim",     "Central"),
    ("Henry Patel",   "North"),
    ("Irene Dubois",  "East"),
    ("Irene Dubois",  "South"),
    ("Jake Wilson",   "West"),
]

PRODUCT_NAMES = [
    "Widget Pro", "Gadget Plus", "Super Donut", "Mega Box",
    "Turbo Pack", "Nano Chip", "Optima Blend", "Delta Force",
    "Echo Drive", "Fusion Kit",
]

CATEGORIES = {
    "Widget Pro":    "Hardware",
    "Gadget Plus":   "Electronics",
    "Super Donut":   "Food & Beverage",
    "Mega Box":      "Packaging",
    "Turbo Pack":    "Accessories",
    "Nano Chip":     "Electronics",
    "Optima Blend":  "Food & Beverage",
    "Delta Force":   "Hardware",
    "Echo Drive":    "Electronics",
    "Fusion Kit":    "Accessories",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _rand_float(lo: float, hi: float, decimals: int = 2) -> float:
    return round(random.uniform(lo, hi), decimals)


# ---------------------------------------------------------------------------
# Table 1: sales_rep_targets
# ---------------------------------------------------------------------------

def build_sales_rep_targets() -> pd.DataFrame:
    """
    Independent quota/territory data per (sales_rep, region).
    quota_usd is set by management discretion — NOT a function of past revenue.
    """
    TERRITORY_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
    MANAGERS = ["Sandra Osei", "Luis Mendes", "Priya Sharma", "Tom Becker"]
    REVIEW_SCORES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    rows = []
    for sales_rep, region in REP_REGION_PAIRS:
        rows.append({
            "sales_rep":          sales_rep,
            "region":             region,
            # quota is an independently set management target (USD)
            "quota_usd":          _rand_float(120_000, 600_000, 0),
            "fte_headcount":      random.randint(1, 6),
            "territory_tier":     random.choice(TERRITORY_TIERS),
            "manager":            random.choice(MANAGERS),
            "hired_date":         _rand_date(date(2015, 1, 1), date(2023, 6, 30)).isoformat(),
            "last_review_score":  random.choice(REVIEW_SCORES),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 2: product_metrics
# ---------------------------------------------------------------------------

def build_product_metrics() -> pd.DataFrame:
    """
    Independent product catalogue / cost data per product.
    unit_cost_usd is a sourcing cost — NOT derived from sales revenue.
    """
    LIFECYCLE_STAGES = ["Intro", "Growth", "Maturity", "Decline"]
    SUPPLIERS = [
        "Apex Manufacturing", "BlueStar Sourcing", "CoreTech Ltd",
        "Delta Imports", "EverGreen Supply",
    ]

    rows = []
    for product_name in PRODUCT_NAMES:
        sku_num = random.randint(10000, 99999)
        rows.append({
            "product_name":       product_name,
            "category":           CATEGORIES[product_name],
            # sourcing cost set by procurement — independent of sales data
            "unit_cost_usd":      _rand_float(2.50, 85.00, 2),
            "launch_year":        random.randint(2012, 2022),
            "lifecycle_stage":    random.choice(LIFECYCLE_STAGES),
            "supplier":           random.choice(SUPPLIERS),
            "sku":                f"SKU-{sku_num}",
            # reorder point set by warehouse ops — independent of sales data
            "reorder_point_units": random.randint(50, 500),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate independent test Excel files.")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Output directory (default: <project_root>/data)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_path = out_dir / "sales_rep_targets.xlsx"
    products_path = out_dir / "product_metrics.xlsx"

    df_targets = build_sales_rep_targets()
    df_targets.to_excel(targets_path, index=False)
    print(f"\u2713 Written: {targets_path}  ({len(df_targets)} rows)")
    print(f"  Columns: {list(df_targets.columns)}")

    df_products = build_product_metrics()
    df_products.to_excel(products_path, index=False)
    print(f"\u2713 Written: {products_path}  ({len(df_products)} rows)")
    print(f"  Columns: {list(df_products.columns)}")

    print("\nUpload both files via the agent:")
    print(f"  load file at data/sales_rep_targets.xlsx as sales_rep_targets")
    print(f"  load file at data/product_metrics.xlsx as product_metrics")


if __name__ == "__main__":
    main()
