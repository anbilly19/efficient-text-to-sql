"""Detect VJ (prior-year) and % Ver. (delta) column pairs."""
from __future__ import annotations

import re

# Patterns that mark a column as a prior-year companion
_VJ_PATTERNS = [
    re.compile(r"\bVJ\b"),           # standalone VJ  e.g. "Umsatz VJ 52 W bis 29/03/26"
    re.compile(r"_VJ$", re.I),
    re.compile(r"\(VJ\)", re.I),
    re.compile(r"Vorjahr", re.I),
]

# Patterns that mark a column as a delta / percent-change
_DELTA_PATTERNS = [
    re.compile(r"%\s*Ver\.?", re.I),   # "% Ver." or "% Ver"
    re.compile(r"Veränderung", re.I),
    re.compile(r"_delta$", re.I),
    re.compile(r"\(delta\)", re.I),
]


def _strip_marker(col_name: str) -> str:
    """Remove the VJ / % Ver. marker token and normalise internal whitespace.

    Approach: substitute the marker with a single space, then collapse all
    runs of whitespace to one space and strip the edges.  This preserves
    the rest of the column name (including period strings like
    '52 W bis 29/03/26') exactly as they appear in the base column.
    """
    name = col_name
    for pat in _VJ_PATTERNS + _DELTA_PATTERNS:
        name = pat.sub(" ", name)
    # Collapse multiple spaces -> single space, strip leading/trailing
    return re.sub(r" {2,}", " ", name).strip()


def detect_suffix_pairs(columns: list[str]) -> dict[str, tuple[str, str]]:
    """
    Returns:
        { derived_col: (base_col, edge_type) }
        where edge_type is "PRIOR_PERIOD_OF" or "DELTA_OF"
    """
    col_set = set(columns)
    result: dict[str, tuple[str, str]] = {}

    for col in columns:
        is_vj    = any(p.search(col) for p in _VJ_PATTERNS)
        is_delta = any(p.search(col) for p in _DELTA_PATTERNS)

        if not (is_vj or is_delta):
            continue

        base = _strip_marker(col)
        if base and base in col_set:
            edge_type = "PRIOR_PERIOD_OF" if is_vj else "DELTA_OF"
            result[col] = (base, edge_type)

    return result
