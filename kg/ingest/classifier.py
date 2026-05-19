"""Classify each column as Metric or Dimension based on dtype + stats."""
from __future__ import annotations

from kg.models import NodeType

_METRIC_DTYPES = {'int64', 'int32', 'float64', 'float32', 'Float64', 'Int64'}
_FLOAT_DTYPES  = {'float64', 'float32', 'Float64'}

# A column is treated as a categorical Dimension when BOTH conditions hold:
#   1. absolute unique count is small (avoids misclassifying on tiny test sets)
#   2. unique ratio is low (avoids misclassifying on large real datasets)
_MAX_CATEGORICAL_UNIQUE = 20
_MAX_CATEGORICAL_RATIO  = 0.10   # n_unique / row_count


def classify_column(col_name: str, col_meta: dict) -> NodeType:
    """Return NodeType.METRIC or NodeType.DIMENSION."""
    dtype    = col_meta.get('dtype', 'object')
    n_unique = col_meta.get('n_unique', 0)
    row_count = max(col_meta.get('_row_count', 1), 1)

    # datetime → always Dimension
    if col_meta.get('likely_datetime') or 'date' in dtype:
        return NodeType.DIMENSION

    # non-numeric → Dimension
    if dtype not in _METRIC_DTYPES:
        return NodeType.DIMENSION

    # Categorical guard: small absolute count AND low ratio → Dimension
    ratio = n_unique / row_count
    if n_unique <= _MAX_CATEGORICAL_UNIQUE and ratio <= _MAX_CATEGORICAL_RATIO:
        return NodeType.DIMENSION

    return NodeType.METRIC
