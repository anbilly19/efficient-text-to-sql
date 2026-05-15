"""
Statistical profiler.

Derives column roles, statistics, period/YoY structure purely from data
distributions and naming patterns.  No domain-specific strings anywhere.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column role detection — driven entirely by statistical signatures
# ---------------------------------------------------------------------------

# Patterns that suggest a column contains period-of-time labels.
# Expressed as Unicode-normalised lowercase regex fragments.
_PERIOD_VALUE_PATTERNS = [
    r"\d{1,2}\s*[wmqjyм]\b",          # "52 W", "4 W", "12 M", "1 J"
    r"(last|letzte|prev)\s*\d",        # "last 12", "letzte 52"
    r"\d{4}[-/]\d{2}[-/]\d{2}",        # ISO date
    r"(q[1-4]|q\s+[1-4])\s*\d{4}",    # Q1 2025
    r"\d{4}\s*(fy|gj|yy)",             # fiscal year marker
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"jan|feb|mär|apr|mai|jun|jul|aug|sep|okt|nov|dez)",
]
_PERIOD_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _PERIOD_VALUE_PATTERNS]

# Phrases in column names that strongly suggest YoY / prior-year values.
# These are structural patterns, not domain terms.
_YOY_NAME_HINTS = [
    r"vs\.?\s*(vj|py|lj|prev|prior|last.year|vorjahr)",
    r"(yoy|y[-_]o[-_]y|year.over.year|jährlich.?veränder)",
    r"(%|pct|prozent).*(ver|change|delta|diff|grow)",
    r"(ver|change|delta|diff|growth).*(%.?|pct|prozent)",
    r"\bvj\b",
]
_YOY_NAME_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _YOY_NAME_HINTS]

# Prior-period companion: column that carries the raw metric value for prior year.
# Distinguished from yoy_delta (which carries the *change*) by not having % / Ver.
_PRIOR_NAME_HINTS = [
    r"\bvj\b(?!.*(%|ver|change|delta))",   # VJ but NOT followed by change indicators
    r"(prior|prev|last.year|vorjahr)\s*(value|val|metric)?$",
    r"(py|lj)\s*$",
]
_PRIOR_NAME_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _PRIOR_NAME_HINTS]


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_rate: float
    n_unique: int
    min_val: Any = None
    max_val: Any = None
    mean_val: float | None = None
    std_val: float | None = None
    top_values: list = field(default_factory=list)
    column_role: str = "unknown"
    role_confidence: float = 0.0


def _norm(text: str) -> str:
    """Unicode-normalise + lowercase for consistent matching."""
    return unicodedata.normalize("NFC", str(text)).lower()


def _detect_period_column(col_name: str, series: pd.Series) -> tuple[bool, float]:
    """Return (is_period, confidence) based on values, not column name alone."""
    if series.dtype not in (object, "string", "str"):
        return False, 0.0
    sample = series.dropna().head(100).astype(str)
    if len(sample) == 0:
        return False, 0.0
    hits = sum(
        1 for v in sample
        if any(pat.search(_norm(v)) for pat in _PERIOD_COMPILED)
    )
    confidence = hits / len(sample)
    return confidence > 0.5, confidence


def _detect_yoy_delta(col_name: str, series: pd.Series) -> tuple[bool, float]:
    """
    YoY delta columns have:
      - Numeric dtype
      - Name matches a structural change-phrase
      - Values centred near 0 with a tail that can go to -100 (%)
    """
    if series.dtype == object:
        return False, 0.0
    name_hit = any(pat.search(_norm(col_name)) for pat in _YOY_NAME_COMPILED)
    if not name_hit:
        return False, 0.0
    vals = series.dropna()
    if len(vals) < 10:
        return name_hit, 0.5
    # Statistical signature: near-zero median, lower bound close to -100
    near_zero_median = abs(float(vals.median())) < 50
    has_negative = float(vals.min()) < -1
    confidence = 0.5 + (0.25 if near_zero_median else 0) + (0.25 if has_negative else 0)
    return True, confidence


def _detect_prior_period(col_name: str) -> tuple[bool, float]:
    hit = any(pat.search(_norm(col_name)) for pat in _PRIOR_NAME_COMPILED)
    return hit, 0.85 if hit else 0.0


def _detect_dimension(col_name: str, series: pd.Series, total_rows: int) -> tuple[bool, float]:
    if series.dtype not in (object, "string", "str"):
        return False, 0.0
    cardinality_ratio = series.nunique() / max(total_rows, 1)
    low_cardinality = cardinality_ratio < 0.1
    return low_cardinality, 1.0 - cardinality_ratio if low_cardinality else 0.0


def profile_dataframe(df: pd.DataFrame) -> list[ColumnProfile]:
    """
    Profile every column in a DataFrame.
    Assigns a role to each column: dimension | metric | yoy_delta | prior_period | period
    No domain knowledge used — purely statistical + structural.
    """
    profiles = []
    n = len(df)

    for col in df.columns:
        series = df[col]
        null_rate = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))

        # Basic stats
        min_val = max_val = mean_val = std_val = None
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean):
                min_val = float(clean.min())
                max_val = float(clean.max())
                mean_val = float(clean.mean())
                std_val = float(clean.std())
        else:
            vals = series.dropna().astype(str)
            if len(vals):
                min_val = str(vals.min())
                max_val = str(vals.max())

        # Top values by frequency
        top_values = (
            series.value_counts(dropna=True).head(5).index.tolist()
        )
        top_values = [str(v) for v in top_values]

        # Role detection — ordered by specificity
        role = "unknown"
        confidence = 0.0

        is_period, conf_p = _detect_period_column(col, series)
        if is_period:
            role, confidence = "period", conf_p
        else:
            is_yoy, conf_y = _detect_yoy_delta(col, series)
            if is_yoy:
                role, confidence = "yoy_delta", conf_y
            else:
                is_prior, conf_pr = _detect_prior_period(col)
                if is_prior:
                    role, confidence = "prior_period", conf_pr
                else:
                    is_dim, conf_d = _detect_dimension(col, series, n)
                    if is_dim:
                        role, confidence = "dimension", conf_d
                    elif pd.api.types.is_numeric_dtype(series):
                        role, confidence = "metric", 0.9
                    else:
                        role, confidence = "dimension", 0.5

        profiles.append(ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            null_rate=null_rate,
            n_unique=n_unique,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
            top_values=top_values,
            column_role=role,
            role_confidence=confidence,
        ))

    return profiles


# ---------------------------------------------------------------------------
# Period / grain inference — completely format-agnostic
# ---------------------------------------------------------------------------

_GRAIN_SIGNATURES: list[tuple[list[str], str]] = [
    # patterns to match in period labels → grain label
    ([r"\b52\s*w\b", r"\b1\s*j\b", r"\b12\s*m\b", r"annual", r"jährlich"], "annual (52-week rolling)"),
    ([r"\b26\s*w\b", r"\b6\s*m\b", r"semi.?annual", r"halbjahr"], "semi-annual (26-week)"),
    ([r"\b13\s*w\b", r"\b3\s*m\b", r"quarter", r"quartal"], "quarterly (13-week)"),
    ([r"\b4\s*w\b", r"\bmonth", r"monat"], "monthly (4-week)"),
    ([r"\b1\s*w\b", r"week", r"woche"], "weekly"),
    ([r"\bday\b", r"\btag\b"], "daily"),
]


def infer_grain(period_values: list[str]) -> str:
    """Infer time grain from actual period label strings."""
    combined = " ".join(str(v) for v in period_values).lower()
    for patterns, label in _GRAIN_SIGNATURES:
        for pat in patterns:
            if re.search(pat, combined, re.IGNORECASE):
                return label
    return "unknown"


def split_cy_py_labels(period_values: list[str]) -> tuple[str | None, str | None]:
    """
    Heuristic: the "current year" label tends to be the most recent date.
    Works by extracting any date-like numeric suffix and sorting.
    """
    if len(period_values) == 0:
        return None, None
    if len(period_values) == 1:
        return period_values[0], None

    def _extract_date_score(label: str) -> int:
        date_m = re.search(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})", label)
        if date_m:
            d, m, y = date_m.groups()
            if len(y) == 2:
                y = "20" + y
            try:
                return int(y) * 10000 + int(m) * 100 + int(d)
            except ValueError:
                pass
        nums = re.findall(r"\d{4,}", label)
        try:
            return int(nums[-1])
        except (IndexError, ValueError):
            return 0

    sorted_labels = sorted(period_values, key=_extract_date_score, reverse=True)
    # Sanity check: if the "VJ" / "prior" marker appears in the top candidate,
    # swap — structural prior-year markers override date ordering
    _PY_MARKER = re.compile(r"\b(vj|py|lj|prior|prev|vorjahr|last\s*year)\b", re.IGNORECASE)
    if len(sorted_labels) >= 2 and _PY_MARKER.search(sorted_labels[0]):
        sorted_labels[0], sorted_labels[1] = sorted_labels[1], sorted_labels[0]
    return sorted_labels[0], sorted_labels[1] if len(sorted_labels) > 1 else None


def profile_to_dict(p: ColumnProfile) -> dict:
    return {
        "column_name": p.name,
        "dtype": p.dtype,
        "null_rate": p.null_rate,
        "n_unique": p.n_unique,
        "min_val": str(p.min_val) if p.min_val is not None else None,
        "max_val": str(p.max_val) if p.max_val is not None else None,
        "mean_val": p.mean_val,
        "std_val": p.std_val,
        "top_values": p.top_values,
        "column_role": p.column_role,
        "role_confidence": p.role_confidence,
    }
