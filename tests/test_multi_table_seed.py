"""
tests/test_multi_table_seed.py
-------------------------------
Unit tests for scripts/generate_product_metrics.py.

Fully self-contained: no database, no agent, no API key.
The generator is called with a real or synthetic source xlsx;
we validate structure, internal consistency, and independence from
the raw sales figures.

Run all:
    pytest tests/test_multi_table_seed.py -v

Run a single class:
    pytest tests/test_multi_table_seed.py::TestSchema -v
    pytest tests/test_multi_table_seed.py::TestValues -v
    pytest tests/test_multi_table_seed.py::TestIndependence -v
    pytest tests/test_multi_table_seed.py::TestFileGeneration -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import generate_product_metrics as gpm


# ---------------------------------------------------------------------------
# Synthetic source fixture — a minimal xlsx with known products/categories
# ---------------------------------------------------------------------------

SYNTH_ROWS = [
    {"Product Name": "Widget A", "Category": "Electronics",  "Unit Price": 99.99,  "Quantity": 3, "Total Revenue": 299.97},
    {"Product Name": "Widget A", "Category": "Electronics",  "Unit Price": 99.99,  "Quantity": 1, "Total Revenue":  99.99},
    {"Product Name": "Gadget B", "Category": "Electronics",  "Unit Price": 49.50,  "Quantity": 2, "Total Revenue":  99.00},
    {"Product Name": "Shirt C",  "Category": "Clothing",     "Unit Price": 29.99,  "Quantity": 5, "Total Revenue": 149.95},
    {"Product Name": "Shirt C",  "Category": "Clothing",     "Unit Price": 29.99,  "Quantity": 2, "Total Revenue":  59.98},
    {"Product Name": "Book D",   "Category": "Books",        "Unit Price": 14.99,  "Quantity": 4, "Total Revenue":  59.96},
]
EXPECTED_PAIRS = {
    ("Widget A", "Electronics"),
    ("Gadget B", "Electronics"),
    ("Shirt C",  "Clothing"),
    ("Book D",   "Books"),
}


@pytest.fixture(scope="module")
def src_xlsx(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "sales.xlsx"
    pd.DataFrame(SYNTH_ROWS).to_excel(p, index=False)
    return p


@pytest.fixture(scope="module")
def df(src_xlsx, tmp_path_factory) -> pd.DataFrame:
    out = tmp_path_factory.mktemp("out") / "product_metrics.xlsx"
    return gpm.generate(src_xlsx, out)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_columns_present(self, df):
        assert list(df.columns) == gpm.COLUMNS

    def test_no_nulls(self, df):
        assert df.isnull().sum().sum() == 0

    def test_product_category_is_pk(self, df):
        assert not df.duplicated(subset=["product", "category"]).any()

    def test_correct_pairs_extracted(self, df):
        found = set(zip(df["product"], df["category"]))
        assert found == EXPECTED_PAIRS


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

class TestValues:
    def test_list_price_positive(self, df):
        assert (df["list_price"] > 0).all()

    def test_prior_units_positive(self, df):
        assert (df["prior_units"] > 0).all()

    def test_prior_orders_positive(self, df):
        assert (df["prior_orders"] > 0).all()

    def test_prior_revenue_positive(self, df):
        assert (df["prior_revenue"] > 0).all()

    def test_avg_order_value_consistent(self, df):
        """avg_order_value == prior_revenue / prior_orders within 1 cent."""
        computed = (df["prior_revenue"] / df["prior_orders"]).round(2)
        diff = (df["avg_order_value"] - computed).abs()
        assert (diff < 0.02).all(), df[["product", "avg_order_value"]][diff >= 0.02]

    def test_prior_units_gte_prior_orders(self, df):
        assert (df["prior_units"] >= df["prior_orders"]).all()

    def test_list_price_in_category_band(self, df):
        for _, row in df.iterrows():
            lo, hi = gpm._PRICE_BANDS.get(row["category"], gpm._DEFAULT_BAND)
            assert lo <= row["list_price"] <= hi, (
                f"{row['product']} ({row['category']}): "
                f"list_price {row['list_price']} outside [{lo}, {hi}]"
            )


# ---------------------------------------------------------------------------
# Independence — values must NOT equal naive GROUP BY on sales1000
# ---------------------------------------------------------------------------

class TestIndependence:
    """The whole point of Option A: catalogue values differ from sales actuals."""

    def test_list_price_not_equal_to_sales_unit_price(self, df, src_xlsx):
        """list_price should not equal the avg unit price from the sales file."""
        raw = pd.read_excel(src_xlsx)
        raw = gpm._normalise(raw)
        # need unit_price col — map from source
        raw = raw.rename(columns={c: "unit_price" for c in raw.columns
                                   if c.strip().lower() in ("unit price", "price")})
        if "unit_price" not in raw.columns:
            pytest.skip("unit_price column not found in synthetic source")
        avg_prices = (
            raw.groupby(["product", "category"])["unit_price"]
            .mean().round(2).reset_index()
            .rename(columns={"unit_price": "sales_avg_price"})
        )
        merged = df.merge(avg_prices, on=["product", "category"])
        # At least some rows must differ (they're from independent RNG)
        diffs = (merged["list_price"] != merged["sales_avg_price"]).sum()
        assert diffs > 0, "list_price is identical to sales avg price for all products"

    def test_reproducible(self, src_xlsx, tmp_path):
        """Running generate() twice with the same src produces identical output."""
        out1 = tmp_path / "pm1.xlsx"
        out2 = tmp_path / "pm2.xlsx"
        df1 = gpm.generate(src_xlsx, out1)
        df2 = gpm.generate(src_xlsx, out2)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                       df2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

class TestFileGeneration:
    def test_writes_xlsx(self, src_xlsx, tmp_path):
        out = tmp_path / "pm.xlsx"
        gpm.generate(src_xlsx, out)
        assert out.exists() and out.stat().st_size > 0

    def test_readable_back(self, src_xlsx, tmp_path):
        out = tmp_path / "pm.xlsx"
        gpm.generate(src_xlsx, out)
        loaded = pd.read_excel(out)
        assert list(loaded.columns) == gpm.COLUMNS

    def test_creates_parent_dirs(self, src_xlsx, tmp_path):
        out = tmp_path / "a" / "b" / "pm.xlsx"
        gpm.generate(src_xlsx, out)
        assert out.exists()

    def test_missing_src_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gpm.generate(tmp_path / "nonexistent.xlsx", tmp_path / "out.xlsx")
