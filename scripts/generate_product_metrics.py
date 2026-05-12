"""
scripts/generate_product_metrics.py
-------------------------------------
Generates data/product_metrics.xlsx as a fully self-contained, hardcoded
product-level dimension table.  No database, no sales1000, no derived data.

The rows represent plausible NIQ-style product baselines and are intentionally
independent of sample_sales_1000.xlsx so the join tests are non-trivial
(attainment won't be exactly 1.0 for every product).

Usage:
    python scripts/generate_product_metrics.py
    # -> writes data/product_metrics.xlsx

    python scripts/generate_product_metrics.py --out path/to/output.xlsx

Columns (matches what the agent and verifier expect):
    product         VARCHAR   -- join key to sales1000."Product Name"
    category        VARCHAR   -- join key to sales1000."Category"
    total_revenue   DOUBLE    -- historical baseline total revenue
    total_units     INTEGER   -- historical units sold
    order_count     INTEGER   -- number of historical orders
    avg_unit_price  DOUBLE    -- historical average unit price
    avg_order_value DOUBLE    -- historical average revenue per order
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Static product catalogue
# Values are plausible baselines; deliberately offset from what sales1000
# will produce so join queries yield interesting, non-trivial results.
# ---------------------------------------------------------------------------

ROWS: list[dict] = [
    # Electronics
    {"product": "Laptop",          "category": "Electronics", "total_revenue": 182_400.00, "total_units": 608,  "order_count": 304, "avg_unit_price": 299.50, "avg_order_value": 599.00},
    {"product": "Smartphone",      "category": "Electronics", "total_revenue": 156_750.00, "total_units": 1045, "order_count": 418, "avg_unit_price": 149.75, "avg_order_value": 374.88},
    {"product": "Tablet",          "category": "Electronics", "total_revenue":  98_560.00, "total_units": 616,  "order_count": 308, "avg_unit_price": 159.95, "avg_order_value": 319.87},
    {"product": "Headphones",      "category": "Electronics", "total_revenue":  47_250.00, "total_units": 1575, "order_count": 525, "avg_unit_price":  29.95, "avg_order_value":  90.00},
    {"product": "Keyboard",        "category": "Electronics", "total_revenue":  38_400.00, "total_units": 960,  "order_count": 480, "avg_unit_price":  39.95, "avg_order_value":  80.00},
    {"product": "Mouse",           "category": "Electronics", "total_revenue":  19_800.00, "total_units": 1980, "order_count": 660, "avg_unit_price":   9.99, "avg_order_value":  30.00},
    {"product": "Monitor",         "category": "Electronics", "total_revenue":  73_500.00, "total_units": 490,  "order_count": 245, "avg_unit_price": 149.00, "avg_order_value": 300.00},
    {"product": "Webcam",          "category": "Electronics", "total_revenue":  14_400.00, "total_units": 720,  "order_count": 360, "avg_unit_price":  19.95, "avg_order_value":  40.00},
    {"product": "USB Hub",         "category": "Electronics", "total_revenue":   8_250.00, "total_units": 1650, "order_count": 550, "avg_unit_price":   4.99, "avg_order_value":  15.00},
    {"product": "Smart Speaker",   "category": "Electronics", "total_revenue":  31_200.00, "total_units": 780,  "order_count": 390, "avg_unit_price":  39.95, "avg_order_value":  80.00},
    # Clothing
    {"product": "T-Shirt",         "category": "Clothing",    "total_revenue":  22_500.00, "total_units": 2500, "order_count": 500, "avg_unit_price":   8.99, "avg_order_value":  45.00},
    {"product": "Jeans",           "category": "Clothing",    "total_revenue":  48_600.00, "total_units": 1080, "order_count": 360, "avg_unit_price":  44.95, "avg_order_value": 135.00},
    {"product": "Jacket",          "category": "Clothing",    "total_revenue":  67_200.00, "total_units": 560,  "order_count": 280, "avg_unit_price": 119.95, "avg_order_value": 240.00},
    {"product": "Sneakers",        "category": "Clothing",    "total_revenue":  81_000.00, "total_units": 900,  "order_count": 300, "avg_unit_price":  89.95, "avg_order_value": 270.00},
    {"product": "Dress",           "category": "Clothing",    "total_revenue":  36_400.00, "total_units": 520,  "order_count": 260, "avg_unit_price":  69.95, "avg_order_value": 140.00},
    {"product": "Hoodie",          "category": "Clothing",    "total_revenue":  29_700.00, "total_units": 990,  "order_count": 330, "avg_unit_price":  29.95, "avg_order_value":  90.00},
    {"product": "Socks",           "category": "Clothing",    "total_revenue":   6_300.00, "total_units": 3150, "order_count": 630, "avg_unit_price":   1.99, "avg_order_value":  10.00},
    {"product": "Belt",            "category": "Clothing",    "total_revenue":  11_250.00, "total_units": 750,  "order_count": 375, "avg_unit_price":  14.95, "avg_order_value":  30.00},
    # Home & Garden
    {"product": "Coffee Maker",    "category": "Home & Garden", "total_revenue":  54_000.00, "total_units": 600, "order_count": 300, "avg_unit_price":  89.95, "avg_order_value": 180.00},
    {"product": "Blender",         "category": "Home & Garden", "total_revenue":  28_800.00, "total_units": 720, "order_count": 360, "avg_unit_price":  39.95, "avg_order_value":  80.00},
    {"product": "Toaster",         "category": "Home & Garden", "total_revenue":  16_500.00, "total_units": 825, "order_count": 275, "avg_unit_price":  19.95, "avg_order_value":  60.00},
    {"product": "Vacuum Cleaner",  "category": "Home & Garden", "total_revenue":  87_500.00, "total_units": 500, "order_count": 250, "avg_unit_price": 174.95, "avg_order_value": 350.00},
    {"product": "Garden Hose",     "category": "Home & Garden", "total_revenue":   9_600.00, "total_units": 480, "order_count": 240, "avg_unit_price":  19.95, "avg_order_value":  40.00},
    {"product": "Lawn Mower",      "category": "Home & Garden", "total_revenue": 124_200.00, "total_units": 414, "order_count": 138, "avg_unit_price": 299.95, "avg_order_value": 900.00},
    # Sports
    {"product": "Yoga Mat",        "category": "Sports",      "total_revenue":  13_500.00, "total_units": 900,  "order_count": 450, "avg_unit_price":  14.95, "avg_order_value":  30.00},
    {"product": "Dumbbell Set",    "category": "Sports",      "total_revenue":  42_000.00, "total_units": 840,  "order_count": 280, "avg_unit_price":  49.95, "avg_order_value": 150.00},
    {"product": "Running Shoes",   "category": "Sports",      "total_revenue":  72_000.00, "total_units": 800,  "order_count": 400, "avg_unit_price":  89.95, "avg_order_value": 180.00},
    {"product": "Bicycle",         "category": "Sports",      "total_revenue": 198_000.00, "total_units": 440,  "order_count": 220, "avg_unit_price": 449.95, "avg_order_value": 900.00},
    {"product": "Tennis Racket",   "category": "Sports",      "total_revenue":  18_900.00, "total_units": 630,  "order_count": 315, "avg_unit_price":  29.95, "avg_order_value":  60.00},
    # Books
    {"product": "Python Cookbook", "category": "Books",       "total_revenue":   8_750.00, "total_units": 1250, "order_count": 625, "avg_unit_price":   6.99, "avg_order_value":  14.00},
    {"product": "Data Science 101","category": "Books",       "total_revenue":  10_200.00, "total_units": 1020, "order_count": 510, "avg_unit_price":  9.99,  "avg_order_value":  20.00},
    {"product": "ML Handbook",     "category": "Books",       "total_revenue":  12_600.00, "total_units": 700,  "order_count": 350, "avg_unit_price":  17.99, "avg_order_value":  36.00},
]


COLUMNS = [
    "product", "category",
    "total_revenue", "total_units", "order_count",
    "avg_unit_price", "avg_order_value",
]


def generate(out_path: str | Path = "data/product_metrics.xlsx") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(ROWS, columns=COLUMNS)
    # Round floats to 2dp for clean display in Excel
    for col in ("total_revenue", "avg_unit_price", "avg_order_value"):
        df[col] = df[col].round(2)

    df.to_excel(out_path, index=False)
    print(f"Written {len(df)} rows -> {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data/product_metrics.xlsx (standalone, no DB required)"
    )
    parser.add_argument(
        "--out",
        default="data/product_metrics.xlsx",
        help="Output path (default: data/product_metrics.xlsx)",
    )
    args = parser.parse_args()
    generate(args.out)
