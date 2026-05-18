"""Reusable helpers for KG ingestion tests (Phase 1, 2, ...).

Import pattern
--------------
    from tests.helpers.kg_fixtures import make_excel, make_niq_style_excel

All DataFrame shapes here mirror the synthetic NIQ-style petfood file used
across Phase 1 and Phase 2 test suites.
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Minimal NIQ-style Excel (used in Phase 1 + Phase 2)
# ---------------------------------------------------------------------------

_ROWS = 30  # enough to trigger header detection and period parsing

#: Column set shared by both Phase 1 and Phase 2 tests.
BASE_COLUMNS: dict[str, list] = {
    "Marke":                                        ["ANIMONDA", "MJAMJAM"] * (_ROWS // 2),
    "Umsatz 52 W bis 29/03/26":                     [100.0, 200.0] * (_ROWS // 2),
    "Umsatz VJ 52 W bis 29/03/26":                  [90.0, 180.0] * (_ROWS // 2),
    "Umsatz % Ver. 52 W bis 29/03/26":              [11.1, 11.1] * (_ROWS // 2),
    "Menge 52 W bis 29/03/26":                      [10, 20] * (_ROWS // 2),
    "Penetration (%) 52 W bis 29/03/26":            [30.0, 40.0] * (_ROWS // 2),
    "K\u00e4uferhaushalte 52 W bis 29/03/26":        [500, 600] * (_ROWS // 2),
    "Unknown Metric XYZ 52 W bis 29/03/26":         [1.0, 2.0] * (_ROWS // 2),
}


def make_excel(path: str, extra_columns: dict | None = None) -> None:
    """Write a synthetic NIQ-style .xlsx to *path*.

    Args:
        path:          Destination file path (must end in .xlsx).
        extra_columns: Optional dict of {col_name: list_of_values} to append.
    """
    data = dict(BASE_COLUMNS)
    if extra_columns:
        data.update(extra_columns)
    pd.DataFrame(data).to_excel(path, index=False)


def make_niq_style_excel(
    path: str,
    *,
    n_rows: int = _ROWS,
    brands: list[str] | None = None,
) -> None:
    """Parameterised variant — useful for tests that need different brand lists
    or row counts without overriding the whole column set."""
    brands = brands or ["ANIMONDA", "MJAMJAM"]
    repeats = (n_rows + len(brands) - 1) // len(brands)  # ceil div
    brand_col = (brands * repeats)[:n_rows]

    df = pd.DataFrame({
        "Marke":                                 brand_col,
        "Umsatz 52 W bis 29/03/26":              [100.0] * n_rows,
        "Umsatz VJ 52 W bis 29/03/26":           [90.0] * n_rows,
        "Umsatz % Ver. 52 W bis 29/03/26":       [11.1] * n_rows,
        "Menge 52 W bis 29/03/26":               [10] * n_rows,
        "Penetration (%) 52 W bis 29/03/26":     [30.0] * n_rows,
        "K\u00e4uferhaushalte 52 W bis 29/03/26": [500] * n_rows,
    })
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# Shared monkeypatch helper (call inside a test, not a fixture)
# ---------------------------------------------------------------------------

def noop_llm_propose(monkeypatch) -> None:
    """Patch concept_mapper._llm_propose to return {} (no API call)."""
    monkeypatch.setattr(
        "kg.ingest.concept_mapper._llm_propose",
        lambda cols: {},
    )
