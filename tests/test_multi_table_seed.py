"""
tests/test_multi_table_seed.py
-------------------------------
Unit tests for scripts/generate_product_metrics.py.

Tests are fully self-contained: no database, no agent, no API key, no
sales1000 file required.  The generator creates the DataFrame in memory;
we validate its structure and values directly.

Run the whole file:
    pytest tests/test_multi_table_seed.py -v

Run a single class:
    pytest tests/test_multi_table_seed.py::TestSchema -v

Run a single test:
    pytest tests/test_multi_table_seed.py::TestValues::test_all_revenue_positive -v

Run with stdout:
    pytest tests/test_multi_table_seed.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import generate_product_metrics as gpm


# ---------------------------------------------------------------------------
# Fixture: build the DataFrame once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.DataFrame(gpm.ROWS, columns=gpm.COLUMNS)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    """Run: pytest tests/test_multi_table_seed.py::TestSchema -v"""

    EXPECTED_COLUMNS = [
        "product", "category",
        "total_revenue", "total_units", "order_count",
        "avg_unit_price", "avg_order_value",
    ]

    def test_all_columns_present(self, df):
        assert list(df.columns) == self.EXPECTED_COLUMNS

    def test_no_null_values(self, df):
        nulls = df.isnull().sum()
        assert nulls.sum() == 0, f"Unexpected nulls:\n{nulls[nulls > 0]}"

    def test_product_category_unique(self, df):
        """(product, category) must be the composite PK."""
        dupes = df.duplicated(subset=["product", "category"])
        assert not dupes.any(), (
            f"Duplicate (product, category) rows:\n{df[dupes]}"
        )

    def test_minimum_row_count(self, df):
        """At least 20 products across at least 3 categories."""
        assert len(df) >= 20
        assert df["category"].nunique() >= 3


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

class TestValues:
    """Run: pytest tests/test_multi_table_seed.py::TestValues -v"""

    def test_all_revenue_positive(self, df):
        assert (df["total_revenue"] > 0).all()

    def test_all_units_positive(self, df):
        assert (df["total_units"] > 0).all()

    def test_all_order_counts_positive(self, df):
        assert (df["order_count"] > 0).all()

    def test_avg_unit_price_positive(self, df):
        assert (df["avg_unit_price"] > 0).all()

    def test_avg_order_value_gte_avg_unit_price(self, df):
        """avg_order_value >= avg_unit_price (multi-unit orders drive this up)."""
        assert (df["avg_order_value"] >= df["avg_unit_price"]).all(), (
            df[df["avg_order_value"] < df["avg_unit_price"]][["product", "avg_unit_price", "avg_order_value"]]
        )

    def test_units_gte_order_count(self, df):
        """total_units >= order_count (each order has at least 1 unit)."""
        assert (df["total_units"] >= df["order_count"]).all()

    def test_avg_order_value_consistent(self, df):
        """avg_order_value ≈ total_revenue / order_count (within 1%)."""
        computed = df["total_revenue"] / df["order_count"]
        ratio = (df["avg_order_value"] / computed).clip(lower=0)
        assert ((ratio - 1).abs() < 0.01).all(), (
            "avg_order_value inconsistent with total_revenue / order_count"
        )


# ---------------------------------------------------------------------------
# Specific products
# ---------------------------------------------------------------------------

class TestSpotCheck:
    """Run: pytest tests/test_multi_table_seed.py::TestSpotCheck -v"""

    def _get(self, df: pd.DataFrame, product: str, category: str) -> pd.Series:
        row = df[(df["product"] == product) & (df["category"] == category)]
        assert len(row) == 1, f"Expected exactly 1 row for ({product}, {category})"
        return row.iloc[0]

    def test_keyboard_present(self, df):
        row = self._get(df, "Keyboard", "Electronics")
        assert row["total_revenue"] > 0

    def test_mouse_present(self, df):
        row = self._get(df, "Mouse", "Electronics")
        assert row["avg_unit_price"] > 0

    def test_tshirt_present(self, df):
        row = self._get(df, "T-Shirt", "Clothing")
        assert row["total_units"] >= row["order_count"]

    def test_jeans_present(self, df):
        row = self._get(df, "Jeans", "Clothing")
        assert abs(row["avg_order_value"] / (row["total_revenue"] / row["order_count"]) - 1) < 0.01

    def test_laptop_is_highest_avg_price_in_electronics(self, df):
        elec = df[df["category"] == "Electronics"]
        assert elec.loc[elec["avg_unit_price"].idxmax(), "product"] == "Laptop"

    def test_bicycle_is_highest_revenue_in_sports(self, df):
        sports = df[df["category"] == "Sports"]
        assert sports.loc[sports["total_revenue"].idxmax(), "product"] == "Bicycle"


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

class TestFileGeneration:
    """Run: pytest tests/test_multi_table_seed.py::TestFileGeneration -v"""

    def test_generate_writes_xlsx(self, tmp_path):
        out = tmp_path / "product_metrics.xlsx"
        gpm.generate(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_generated_file_is_readable(self, tmp_path):
        out = tmp_path / "product_metrics.xlsx"
        gpm.generate(out)
        df_loaded = pd.read_excel(out)
        assert list(df_loaded.columns) == gpm.COLUMNS
        assert len(df_loaded) == len(gpm.ROWS)

    def test_generate_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "metrics.xlsx"
        gpm.generate(nested)
        assert nested.exists()
