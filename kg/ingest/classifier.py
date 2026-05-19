"""Classify each column as Metric or Dimension based on dtype + stats."""
from __future__ import annotations

from kg.models import NodeType

# Numeric dtypes that *could* be metrics (floats are almost always metrics).
_METRIC_DTYPES = {'int64', 'int32', 'float64', 'float32', 'Float64', 'Int64'}

# Absolute unique-value ceiling below which a numeric column is treated as a
# categorical Dimension (e.g. year codes, rating scales, category IDs).
# Kept intentionally small so real metric columns (even with few rows in
# synthetic test data) are not wrongly demoted to Dimension.
_MAX_CATEGORICAL_UNIQUE = 10

# Float columns are almost always continuous metrics — only demote if they
# also pass the categorical check.
_FLOAT_DTYPES = {'float64', 'float32', 'Float64'}


def classify_column(col_name: str, col_meta: dict) -> NodeType:
    """Return NodeType.METRIC or NodeType.DIMENSION."""
    dtype = col_meta.get('dtype', 'object')
    n_unique = col_meta.get('n_unique', 0)

    # Explicit datetime → always Dimension
    if col_meta.get('likely_datetime') or 'date' in dtype:
        return NodeType.DIMENSION

    # Non-numeric → Dimension
    if dtype not in _METRIC_DTYPES:
        return NodeType.DIMENSION

    # Float columns: treat as Metric unless the unique count is suspiciously
    # small AND there are very few distinct values (e.g. a 0/1 flag stored as
    # float).  Threshold: <= 3 unique values.
    if dtype in _FLOAT_DTYPES:
        if n_unique <= 3:
            return NodeType.DIMENSION
        return NodeType.METRIC

    # Integer columns: low absolute cardinality → Dimension (year, code, id)
    if n_unique <= _MAX_CATEGORICAL_UNIQUE:
        return NodeType.DIMENSION

    return NodeType.METRIC
