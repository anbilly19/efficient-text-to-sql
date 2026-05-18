"""Classify each column as Metric or Dimension based on dtype + stats."""
from __future__ import annotations

from kg.models import NodeType

# Numeric columns with high cardinality are Metrics.
# String/low-cardinality numeric columns are Dimensions.
_METRIC_DTYPES = {"int64", "int32", "float64", "float32", "Float64", "Int64"}
_MAX_DIMENSION_UNIQUE = 200  # if n_unique > this AND numeric → Metric


def classify_column(col_name: str, col_meta: dict) -> NodeType:
    """Return NodeType.METRIC or NodeType.DIMENSION."""
    dtype = col_meta.get("dtype", "object")
    n_unique = col_meta.get("n_unique", 0)
    row_count = col_meta.get("_row_count", 1)  # injected by pipeline

    # Explicit datetime → always Dimension
    if col_meta.get("likely_datetime") or "date" in dtype:
        return NodeType.DIMENSION

    # Numeric with high cardinality → Metric
    if dtype in _METRIC_DTYPES:
        # Low-cardinality numeric (e.g. year, code) → Dimension
        if n_unique <= _MAX_DIMENSION_UNIQUE and n_unique < row_count * 0.05:
            return NodeType.DIMENSION
        return NodeType.METRIC

    # Everything else (object / category / bool) → Dimension
    return NodeType.DIMENSION
