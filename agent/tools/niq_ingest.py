"""
agent/tools/niq_ingest.py
─────────────────────────
Dynamic ingest pipeline for NIQ-like panel datasets.

Design goals
────────────
- Fully NIQ-agnostic at runtime: no column names are hard-coded here.
  All knowledge is derived by inspecting the actual loaded DataFrame.
- Bridges to schema_lookup.SCHEMA_CATALOG for alias seeding until the
  LLM-based auto-alias generator lands (Phase 1 → Phase 3 migration).
- Produces rich _column_catalog rows (niq_role, description) and
  _table_context (summary, grain, tags) so the orchestrator and SQL-writer
  can reason about CY/PY structure without knowing column names in advance.

Public API
──────────
  load_niq_file(path, dataset_name)  — LangChain @tool, top-level entry point
  detect_niq_structure(df)           — pure function, returns NiqMeta
  classify_niq_columns(df)           — pure function, returns col -> NiqColInfo
  seed_niq_semantic_map(conn, dataset_name, col_info)  — writes _semantic_map rows
  build_niq_table_context(meta)      — returns (summary, grain, tags)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Column role constants
# ---------------------------------------------------------------------------

ROLE_DIMENSION = "dimension"
ROLE_METRIC_CY = "metric_cy"      # current-year metric
ROLE_METRIC_PY = "metric_py"      # prior-year (VJ) baseline
ROLE_YOY_DELTA = "yoy_delta"      # % Veränderung vs. VJ
ROLE_UNKNOWN   = "unknown"

_VJ_SUFFIXES  = (" vj", "_vj", "(vj)")
_YOY_PATTERNS = ("% ver.", "vs. vj", "veränderung")
_METADATA_VALUES = {"market", "markets", "fact", "facts", "universe", "base"}

# Column names that unambiguously identify the real NIQ header row.
_NIQ_ANCHOR_COLS = {"periods", "products", "geographies", "retailers", "demographics"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NiqColInfo:
    role:        str
    description: str = ""
    aliases:     list[str] = field(default_factory=list)


@dataclass
class NiqMeta:
    """Structural metadata extracted from a NIQ-like DataFrame."""
    has_periods_col:   bool       = False
    periods_col:       str        = ""
    cy_period_label:   str        = ""
    py_period_label:   str        = ""
    dimension_cols:    list[str]  = field(default_factory=list)
    metric_cy_cols:    list[str]  = field(default_factory=list)
    metric_py_cols:    list[str]  = field(default_factory=list)
    yoy_delta_cols:    list[str]  = field(default_factory=list)
    metadata_row_mask: list[bool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preamble detection
# ---------------------------------------------------------------------------

def _find_header_row(file_path: Path) -> int:
    """
    Scan the first 30 rows of *file_path* (Excel or CSV) and return the
    0-based row index that contains the real column headers.

    NIQ exports embed copyright/export-date text above the data table.
    The real header is the first row where >= 2 of the known NIQ anchor
    column names appear (case-insensitive).

    Returns 0 if no preamble is detected (safe default).
    """
    suffix = file_path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            raw = pd.read_excel(file_path, header=None, nrows=30)
        else:
            raw = pd.read_csv(file_path, header=None, nrows=30)
    except Exception:
        return 0

    for i, row in raw.iterrows():
        vals = {str(v).strip().lower() for v in row.values if pd.notna(v)}
        if len(vals & _NIQ_ANCHOR_COLS) >= 2:
            return int(i)
    return 0


# ---------------------------------------------------------------------------
# DataFrame cleaning
# ---------------------------------------------------------------------------

def _sanitize_df(df: pd.DataFrame, col_info: dict[str, NiqColInfo]) -> pd.DataFrame:
    """
    Clean *df* so it is safe to write as Parquet.

      1. Drop all-NaN rows (footer/blank lines).
      2. Drop Unnamed:* columns (Excel header bleed).
      3. Coerce metric/yoy object cols to numeric.
      4. Cast remaining object cols to nullable StringDtype.
    """
    # 1. Drop rows that are entirely NaN
    df = df.dropna(how="all").reset_index(drop=True)

    # 2. Drop Unnamed columns
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # 3 & 4. Fix object dtype columns
    for col in list(df.select_dtypes(include="object").columns):
        role = col_info.get(col, NiqColInfo(role=ROLE_DIMENSION)).role
        if role in (ROLE_METRIC_CY, ROLE_METRIC_PY, ROLE_YOY_DELTA):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        else:
            # Nullable string — maps float NaN → pd.NA, pyarrow writes as null
            df[col] = df[col].where(df[col].notna(), other=None).astype("string")

    return df


# ---------------------------------------------------------------------------
# Pure detection helpers
# ---------------------------------------------------------------------------

def detect_niq_structure(df: pd.DataFrame) -> NiqMeta:
    """
    Inspect *df* and return an NiqMeta describing its structure.

    Detection rules (in priority order):
      1. Periods column: column named 'Periods'/'Period' (case-insensitive),
         or a low-cardinality string column with 2 distinct values both
         matching NIQ period-label patterns.
      2. Metadata rows: rows where ALL dimension columns hold banner/header
         values from _METADATA_VALUES.
    """
    meta = NiqMeta()
    col_info = classify_niq_columns(df)

    meta.dimension_cols = [c for c, i in col_info.items() if i.role == ROLE_DIMENSION]
    meta.metric_cy_cols = [c for c, i in col_info.items() if i.role == ROLE_METRIC_CY]
    meta.metric_py_cols = [c for c, i in col_info.items() if i.role == ROLE_METRIC_PY]
    meta.yoy_delta_cols = [c for c, i in col_info.items() if i.role == ROLE_YOY_DELTA]

    # Detect Periods column by name first
    for col in df.columns:
        if col.strip().lower() in ("periods", "period"):
            meta.has_periods_col = True
            meta.periods_col = col
            break

    # Fallback: low-cardinality string col with NIQ period-label patterns
    if not meta.has_periods_col:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            vals = df[col].dropna().unique()
            if len(vals) == 2 and all(
                "letzte" in str(v).lower() or re.search(r"\d{2}/\d{2}/\d{2}", str(v))
                for v in vals
            ):
                meta.has_periods_col = True
                meta.periods_col = col
                break

    # Extract CY / PY period labels
    if meta.has_periods_col and meta.periods_col in df.columns:
        for v in df[meta.periods_col].dropna().unique():
            vs = str(v)
            if " vj " in vs.lower() or vs.lower().endswith(" vj"):
                meta.py_period_label = vs
            else:
                meta.cy_period_label = vs

    # Metadata row mask
    if meta.dimension_cols:
        def _is_meta_row(row: pd.Series) -> bool:
            vals = [
                str(row[c]).strip().lower()
                for c in meta.dimension_cols
                if c in row.index and pd.notna(row[c]) and str(row[c]) != "<NA>"
            ]
            return not vals or all(v in _METADATA_VALUES or v in ("nan", "<na>") for v in vals)
        meta.metadata_row_mask = df.apply(_is_meta_row, axis=1).tolist()
    else:
        meta.metadata_row_mask = [False] * len(df)

    return meta


def classify_niq_columns(df: pd.DataFrame) -> dict[str, NiqColInfo]:
    """
    Classify every column in *df* into a NIQ semantic role.

    Priority:
      1. YoY delta  — name contains any _YOY_PATTERNS token
      2. Prior year — name ends with any _VJ_SUFFIXES token
      3. Numeric dtype                                → metric_cy
      4. Otherwise                                    → dimension
    """
    try:
        from agent.schema_lookup import SCHEMA_CATALOG as _CATALOG
    except ImportError:
        _CATALOG = {}

    catalog_lower: dict[str, dict] = {k.lower(): v for k, v in _CATALOG.items()}
    result: dict[str, NiqColInfo] = {}

    for col in df.columns:
        col_lo = col.strip().lower()

        if any(pat in col_lo for pat in _YOY_PATTERNS):
            role = ROLE_YOY_DELTA
        elif any(col_lo.endswith(suf) for suf in _VJ_SUFFIXES):
            role = ROLE_METRIC_PY
        elif pd.api.types.is_numeric_dtype(df[col]):
            role = ROLE_METRIC_CY
        else:
            role = ROLE_DIMENSION

        cat_entry   = catalog_lower.get(col_lo, {})
        description = cat_entry.get("description", "") or cat_entry.get("en", "")
        aliases     = list(cat_entry.get("aliases", []))

        result[col] = NiqColInfo(role=role, description=description, aliases=aliases)

    return result


def build_niq_table_context(
    meta: NiqMeta,
    dataset_name: str,
    source_filename: str,
) -> tuple[str, str, list[str]]:
    """Build (summary, grain, tags) for _table_context from NiqMeta."""
    n_dim    = len(meta.dimension_cols)
    n_metric = len(meta.metric_cy_cols)
    n_yoy    = len(meta.yoy_delta_cols) + len(meta.metric_py_cols)

    if meta.has_periods_col and meta.cy_period_label:
        period_desc = f"CY='{meta.cy_period_label}'"
        if meta.py_period_label:
            period_desc += f", PY='{meta.py_period_label}'"
    elif meta.has_periods_col:
        period_desc = "CY vs PY (two-period structure)"
    else:
        period_desc = "single-period"

    summary = (
        f"NIQ panel dataset loaded from '{source_filename}'. "
        f"{n_dim} dimension(s), {n_metric} CY metric(s), {n_yoy} YoY column(s). "
        f"Period: {period_desc}."
    )
    grain = "annual, CY vs PY" if meta.has_periods_col else "annual"
    tags  = ["niq", "panel", "haushalte", "penetration", "purchase", "frequency"]
    dim_lo = [c.lower() for c in meta.dimension_cols]
    if "products"    in dim_lo: tags.append("product")
    if "retailers"   in dim_lo: tags.extend(["retailer", "channel", "shop"])
    if "geographies" in dim_lo: tags.extend(["geography", "market", "region"])

    return summary, grain, tags


# ---------------------------------------------------------------------------
# Semantic map seeding
# ---------------------------------------------------------------------------

def seed_niq_semantic_map(
    conn,
    dataset_name: str,
    col_info: dict[str, NiqColInfo],
) -> int:
    """Upsert alias + canonical-name rows into _semantic_map. Returns row count."""
    rows_written = 0

    def _upsert(term: str, col_name: str, description: str | None) -> None:
        nonlocal rows_written
        term = term.strip().lower()
        if not term:
            return
        try:
            conn.execute(
                """
                INSERT INTO _semantic_map (term, dataset_name, column_name, description)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (term, dataset_name)
                DO UPDATE SET column_name = excluded.column_name,
                              description = excluded.description
                """,
                [term, dataset_name, col_name, description or None],
            )
            rows_written += 1
        except Exception:
            pass

    for col_name, info in col_info.items():
        _upsert(col_name, col_name, info.description)
        for alias in info.aliases:
            _upsert(alias, col_name, info.description)

    return rows_written


# ---------------------------------------------------------------------------
# _column_catalog enrichment
# ---------------------------------------------------------------------------

def _enrich_column_catalog(
    conn,
    dataset_name: str,
    col_info: dict[str, NiqColInfo],
) -> None:
    """Patch _column_catalog rows with NIQ role flags and descriptions."""
    for col_name, info in col_info.items():
        is_metric    = info.role in (ROLE_METRIC_CY, ROLE_METRIC_PY, ROLE_YOY_DELTA)
        is_dimension = info.role == ROLE_DIMENSION
        try:
            conn.execute(
                """
                UPDATE _column_catalog
                SET is_metric    = ?,
                    is_dimension = ?,
                    description  = COALESCE(NULLIF(description, ''), ?)
                WHERE dataset_name = ? AND column_name = ?
                """,
                [is_metric, is_dimension, info.description or None, dataset_name, col_name],
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal: resolve private helpers from flat tools.py without circular import
# ---------------------------------------------------------------------------

def _get_flat_attr(name: str):
    """Retrieve *name* from agent/tools.py loaded under the _agent_tools_flat alias."""
    import sys, importlib.util
    _ALIAS = "_agent_tools_flat"
    if _ALIAS not in sys.modules:
        flat = Path(__file__).parent.parent / "tools.py"
        spec = importlib.util.spec_from_file_location(_ALIAS, flat)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[_ALIAS] = mod
        spec.loader.exec_module(mod)
    return getattr(sys.modules[_ALIAS], name)


# ---------------------------------------------------------------------------
# Main entry-point tool
# ---------------------------------------------------------------------------

@tool
def load_niq_file(path: str, dataset_name: str) -> str:
    """Load a NIQ-format panel dataset (Excel/CSV/Parquet) into DuckDB.

    Handles NIQ exports that embed a copyright/metadata preamble above the
    real data table.  The preamble is auto-detected and skipped.

    Post-processing:
      - Drops Unnamed:* columns and all-NaN rows
      - Sanitizes mixed-type object columns so Parquet write never fails
      - Auto-detects Periods column and CY/PY period labels
      - Classifies every column: dimension / metric_cy / metric_py / yoy_delta
      - Drops header/banner metadata rows embedded in NIQ exports
      - Enriches _column_catalog with role flags and descriptions
      - Seeds _semantic_map with aliases from SCHEMA_CATALOG
      - Sets grain = 'annual, CY vs PY' in _table_context

    Args:
        path: Absolute or relative path to the NIQ source file.
        dataset_name: Name to register as in DuckDB.
    """
    from agent.database import get_connection, index_table_schema
    from agent.db.catalog import upsert_table_context

    _get_parquet_store = _get_flat_attr("_get_parquet_store")
    _PROJECT_ROOT      = _get_flat_attr("_PROJECT_ROOT")

    # ── 1. Resolve file path ─────────────────────────────────────────────
    file_path = Path(path)
    if not file_path.exists():
        file_path = _PROJECT_ROOT / path
    if not file_path.exists():
        return f"ERROR: File not found at '{path}'"
    file_path = file_path.resolve()
    suffix = file_path.suffix.lower()

    # ── 2. Auto-detect preamble and read file ─────────────────────────────
    try:
        if suffix in (".xlsx", ".xls"):
            skip = _find_header_row(file_path)
            df = pd.read_excel(file_path, skiprows=skip)
        elif suffix == ".csv":
            skip = _find_header_row(file_path)
            df = pd.read_csv(file_path, skiprows=skip, low_memory=False)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"ERROR: Unsupported file type '{suffix}'"
    except Exception as exc:
        return f"ERROR reading file: {exc}"

    # ── 3. Classify columns on the raw (post-header) DataFrame ──────────────
    col_info = classify_niq_columns(df)
    meta     = detect_niq_structure(df)

    # ── 4. Drop metadata / banner rows ────────────────────────────────────
    n_raw = len(df)
    if any(meta.metadata_row_mask):
        df = df[[not m for m in meta.metadata_row_mask]].reset_index(drop=True)
    n_clean = len(df)
    dropped = n_raw - n_clean

    # ── 5. Sanitize DataFrame (Unnamed cols, mixed types, all-NaN rows) ──────
    df       = _sanitize_df(df, col_info)
    # Re-classify after sanitization (dtypes may have changed)
    col_info = classify_niq_columns(df)
    meta     = detect_niq_structure(df)
    n_clean  = len(df)

    # ── 6. Write Parquet + register DuckDB view ────────────────────────────
    conn          = get_connection()
    parquet_store = _get_parquet_store()
    parquet_path  = parquet_store / f"{dataset_name}.parquet"
    try:
        parquet_store.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
    except Exception as exc:
        return f"ERROR writing Parquet: {exc}"

    try:
        conn.execute(
            f'CREATE OR REPLACE VIEW "{dataset_name}" AS '
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        )
    except Exception as exc:
        return f"ERROR registering DuckDB view: {exc}"

    # ── 7. Update _data_registry ──────────────────────────────────────────
    try:
        conn.execute(
            """
            INSERT INTO _data_registry
                (dataset_name, parquet_path, source_file, row_count, column_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (dataset_name) DO UPDATE SET
                parquet_path  = excluded.parquet_path,
                source_file   = excluded.source_file,
                row_count     = excluded.row_count,
                column_count  = excluded.column_count,
                ingested_at   = now()
            """,
            [dataset_name, str(parquet_path.resolve()), str(file_path),
             n_clean, len(df.columns)],
        )
    except Exception as exc:
        return f"File loaded but registry update failed: {exc}"

    # ── 8. Index + enrich column catalog ──────────────────────────────────
    try:
        index_table_schema(conn, dataset_name)
    except Exception as exc:
        return f"File loaded but column catalog indexing failed: {exc}"
    _enrich_column_catalog(conn, dataset_name, col_info)

    # ── 9. Seed _semantic_map ─────────────────────────────────────────────
    n_aliases = seed_niq_semantic_map(conn, dataset_name, col_info)

    # ── 10. Write _table_context ───────────────────────────────────────────
    summary, grain, tags = build_niq_table_context(meta, dataset_name, file_path.name)
    upsert_table_context(dataset_name, summary, grain, tags, conn=conn)

    # ── 11. Response ─────────────────────────────────────────────────────
    role_summary = {
        "dimensions": len(meta.dimension_cols),
        "metrics_CY": len(meta.metric_cy_cols),
        "metrics_PY": len(meta.metric_py_cols),
        "yoy_deltas": len(meta.yoy_delta_cols),
    }
    periods_msg = (
        f" Periods: CY='{meta.cy_period_label}' / PY='{meta.py_period_label}'."
        if meta.has_periods_col and meta.cy_period_label
        else " No Periods column detected."
    )
    drop_msg = f" Dropped {dropped} metadata row(s)." if dropped else ""
    skip_msg = f" Skipped {skip} preamble row(s)." if suffix != ".parquet" and skip > 0 else ""

    return (
        f"Successfully loaded NIQ file '{file_path.name}' as '{dataset_name}'. "
        f"{len(df.columns)} columns, {n_clean} data rows."
        f"{skip_msg}"
        f"{drop_msg}"
        f" Column roles: {role_summary}."
        f"{periods_msg}"
        f" Seeded {n_aliases} semantic alias(es)."
    )
