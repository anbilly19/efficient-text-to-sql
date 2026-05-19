"""Reusable helpers for KG ingestion tests (Phase 1, 2, ...)."""
from __future__ import annotations

import random
import pandas as pd

_ROWS = 30
_RNG  = random.Random(42)  # deterministic

# Realistic varied values so metric columns have n_unique >> 10
# (openpyxl round-trips floats faithfully; integers stay int64)
_BRANDS = ['ANIMONDA', 'MJAMJAM'] * (_ROWS // 2)

def _floats(lo: float, hi: float) -> list:
    return [round(_RNG.uniform(lo, hi), 2) for _ in range(_ROWS)]

def _ints(lo: int, hi: int) -> list:
    return [_RNG.randint(lo, hi) for _ in range(_ROWS)]

BASE_COLUMNS: dict[str, list] = {
    'Marke':                                        _BRANDS,
    'Umsatz 52 W bis 29/03/26':                     _floats(80.0, 300.0),
    'Umsatz VJ 52 W bis 29/03/26':                  _floats(70.0, 280.0),
    'Umsatz % Ver. 52 W bis 29/03/26':              _floats(-20.0, 30.0),
    'Menge 52 W bis 29/03/26':                      _ints(5, 50),
    'Penetration (%) 52 W bis 29/03/26':            _floats(10.0, 60.0),
    'K\u00e4uferhaushalte 52 W bis 29/03/26':        _ints(300, 900),
    'Unknown Metric XYZ 52 W bis 29/03/26':         _floats(0.5, 5.0),
}


def make_excel(path: str, extra_columns: dict | None = None) -> None:
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
    brands = brands or ['ANIMONDA', 'MJAMJAM']
    repeats = (n_rows + len(brands) - 1) // len(brands)
    brand_col = (brands * repeats)[:n_rows]
    rng = random.Random(42)

    def f(lo, hi): return [round(rng.uniform(lo, hi), 2) for _ in range(n_rows)]
    def i(lo, hi): return [rng.randint(lo, hi) for _ in range(n_rows)]

    df = pd.DataFrame({
        'Marke':                                 brand_col,
        'Umsatz 52 W bis 29/03/26':              f(80, 300),
        'Umsatz VJ 52 W bis 29/03/26':           f(70, 280),
        'Umsatz % Ver. 52 W bis 29/03/26':       f(-20, 30),
        'Menge 52 W bis 29/03/26':               i(5, 50),
        'Penetration (%) 52 W bis 29/03/26':     f(10, 60),
        'K\u00e4uferhaushalte 52 W bis 29/03/26': i(300, 900),
    })
    df.to_excel(path, index=False)


def noop_llm_propose(monkeypatch) -> None:
    """Patch concept_mapper._llm_propose to return {} (no API call)."""
    monkeypatch.setattr(
        'kg.ingest.concept_mapper._llm_propose',
        lambda cols: {},
    )
