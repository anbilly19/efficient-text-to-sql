#!/usr/bin/env python
"""
scripts/generate_test_data.py

Generates sales_rep_targets.xlsx and product_metrics.xlsx whose key columns
MATCH the actual distinct values in data/sample_sales_1000.xlsx so that JOIN
queries return non-empty results.

If sample_sales_1000.xlsx is not found the script falls back to hardcoded
example values (useful for CI where the real file may not be present).

Usage:
    python scripts/generate_test_data.py
    python scripts/generate_test_data.py --out /tmp/test_data
"""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _rand_float(lo: float, hi: float, decimals: int = 2) -> float:
    return round(random.uniform(lo, hi), decimals)


def _read_sales1000(out_dir: Path) -> pd.DataFrame | None:
    """Try to load sample_sales_1000.xlsx from data/ directory."""
    candidates = [
        ROOT / "data" / "sample_sales_1000.xlsx",
        out_dir / "sample_sales_1000.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_excel(p)
    return None


# ---------------------------------------------------------------------------
# Table 1: sales_rep_targets
# Key columns must match sales1000.sales_rep and sales1000.region exactly.
# ---------------------------------------------------------------------------

def build_sales_rep_targets(sales_reps: list[str], regions: list[str]) -> pd.DataFrame:
    TERRITORY_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
    MANAGERS = ["Sandra Osei", "Luis Mendes", "Priya Sharma", "Tom Becker"]
    REVIEW_SCORES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    # Build (rep, region) pairs: each rep gets 1-2 regions, cover all combos
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for rep in sales_reps:
        n = random.randint(1, min(2, len(regions)))
        for region in random.sample(regions, n):
            if (rep, region) not in seen:
                seen.add((rep, region))
                pairs.append((rep, region))
    # Make sure every region is covered at least once
    for region in regions:
        if not any(r == region for _, r in pairs):
            rep = random.choice(sales_reps)
            if (rep, region) not in seen:
                seen.add((rep, region))
                pairs.append((rep, region))

    rows = []
    for sales_rep, region in pairs:
        rows.append({
            "sales_rep":         sales_rep,
            "region":            region,
            "quota_usd":         _rand_float(120_000, 600_000, 0),
            "fte_headcount":     random.randint(1, 6),
            "territory_tier":    random.choice(TERRITORY_TIERS),
            "manager":           random.choice(MANAGERS),
            "hired_date":        _rand_date(date(2015, 1, 1), date(2023, 6, 30)).isoformat(),
            "last_review_score": random.choice(REVIEW_SCORES),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 2: product_metrics
# Key columns: product_name must match sales1000.product; category must match.
# ---------------------------------------------------------------------------

def build_product_metrics(
    products: list[str],
    product_to_category: dict[str, str],
) -> pd.DataFrame:
    LIFECYCLE_STAGES = ["Intro", "Growth", "Maturity", "Decline"]
    SUPPLIERS = [
        "Apex Manufacturing", "BlueStar Sourcing", "CoreTech Ltd",
        "Delta Imports", "EverGreen Supply",
    ]

    rows = []
    for product_name in products:
        rows.append({
            "product_name":        product_name,
            "category":            product_to_category.get(product_name, "General"),
            "unit_cost_usd":       _rand_float(2.50, 85.00, 2),
            "launch_year":         random.randint(2012, 2022),
            "lifecycle_stage":     random.choice(LIFECYCLE_STAGES),
            "supplier":            random.choice(SUPPLIERS),
            "sku":                 f"SKU-{random.randint(10000, 99999)}",
            "reorder_point_units": random.randint(50, 500),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "data"),
        help="Output directory (default: <project_root>/data)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read actual values from sales1000 ──────────────────────────────────
    df_sales = _read_sales1000(out_dir)
    if df_sales is not None:
        print(f"Reading key values from sample_sales_1000.xlsx ({len(df_sales)} rows)")
        sales_reps   = sorted(df_sales["sales_rep"].dropna().unique().tolist())
        regions      = sorted(df_sales["region"].dropna().unique().tolist())
        products     = sorted(df_sales["product"].dropna().unique().tolist())
        # Build product->category mapping from actual data
        prod_cat = (
            df_sales[["product", "category"]]
            .drop_duplicates()
            .set_index("product")["category"]
            .to_dict()
        )
    else:
        print("WARNING: sample_sales_1000.xlsx not found -- using fallback values")
        sales_reps = [
            "Alice Morgan", "Bob Nguyen", "Carol Smith", "David Lee",
            "Eva Fischer", "Frank Torres", "Grace Kim", "Henry Patel",
            "Irene Dubois", "Jake Wilson",
        ]
        regions  = ["North", "South", "East", "West", "Central"]
        products = [
            "Widget Pro", "Gadget Plus", "Super Donut", "Mega Box",
            "Turbo Pack", "Nano Chip", "Optima Blend", "Delta Force",
            "Echo Drive", "Fusion Kit",
        ]
        prod_cat = {
            "Widget Pro": "Hardware", "Gadget Plus": "Electronics",
            "Super Donut": "Food & Beverage", "Mega Box": "Packaging",
            "Turbo Pack": "Accessories", "Nano Chip": "Electronics",
            "Optima Blend": "Food & Beverage", "Delta Force": "Hardware",
            "Echo Drive": "Electronics", "Fusion Kit": "Accessories",
        }

    print(f"  {len(sales_reps)} sales reps, {len(regions)} regions, {len(products)} products")

    # ── Write files ────────────────────────────────────────────────────────
    targets_path = out_dir / "sales_rep_targets.xlsx"
    products_path = out_dir / "product_metrics.xlsx"

    df_targets = build_sales_rep_targets(sales_reps, regions)
    df_targets.to_excel(targets_path, index=False)
    print(f"✓ Written: {targets_path}  ({len(df_targets)} rows)")
    print(f"  Columns: {list(df_targets.columns)}")

    df_products = build_product_metrics(products, prod_cat)
    df_products.to_excel(products_path, index=False)
    print(f"✓ Written: {products_path}  ({len(df_products)} rows)")
    print(f"  Columns: {list(df_products.columns)}")


if __name__ == "__main__":
    main()
