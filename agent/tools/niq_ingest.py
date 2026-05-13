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
  seed_niq_semantic_map(conn, dataset_name, df)  — writes _semantic_map rows
  build_niq_table_context(meta)      — returns (summary, grain, tags)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Column role constants
# ---------------------------------------------------------------------------

ROLE_DIMENSION = "dimension"
ROLE_METRIC_CY = "metric_cy"      # current-year metric
ROLE_METRIC_PY = "metric_py"      # prior-year (VJ) baseline
ROLE_YOY_DELTA = "yoy_delta"       # % Veränderung vs. VJ
ROLE_UNKNOWN   = "unknown"

# Suffixes / patterns that identify prior-year and YoY-delta columns.
_VJ_SUFFIXES     = (" vj", "_vj", "(vj)")             # prior year
_YOY_PATTERNS    = ("% ver.", "vs. vj", "veränderung") # YoY delta
_METADATA_VALUES = {
    # Values that appear in dimension columns when a row is a header/banner
    # rather than a real data row in NIQ exports.
    "market", "markets", "fact", "facts", "universe", "base",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NiqColInfo:
    role:        str          # one of the ROLE_* constants
    description: str = ""
    aliases:     list[str] = field(default_factory=list)


@dataclass
class NiqMeta:
    """Structural metadata extracted from a NIQ-like DataFrame."""
    has_periods_col:   bool          = False
    periods_col:       str           = ""
    cy_period_label:   str           = ""   # e.g. 'Letzte 12 M - 52 W bis 28/12/25'
    py_period_label:   str           = ""   # e.g. 'Letzte 12 M VJ - 52 W bis 29/12/24'
    dimension_cols:    list[str]     = field(default_factory=list)
    metric_cy_cols:    list[str]     = field(default_factory=list)
    metric_py_cols:    list[str]     = field(default_factory=list)
    yoy_delta_cols:    list[str]     = field(default_factory=list)
    metadata_row_mask: list[bool]    = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure detection helpers
# ---------------------------------------------------------------------------

def detect_niq_structure(df: pd.DataFrame) -> NiqMeta:
    """
    Inspect *df* and return an NiqMeta describing its structure.

    Detection rules (in priority order):
      1. Periods column: VARCHAR column named 'Periods' (case-insensitive),
         or any VARCHAR column with exactly 2 distinct non-null values where
         both look like 'Letzte 12 M' period strings.
      2. Metadata rows: rows where ALL dimension columns are either null or
         contain a known banner/header value.
    """
    meta = NiqMeta()
    col_info = classify_niq_columns(df)

    meta.dimension_cols = [c for c, i in col_info.items() if i.role == ROLE_DIMENSION]
    meta.metric_cy_cols = [c for c, i in col_info.items() if i.role == ROLE_METRIC_CY]
    meta.metric_py_cols = [c for c, i in col_info.items() if i.role == ROLE_METRIC_PY]
    meta.yoy_delta_cols = [c for c, i in col_info.items() if i.role == ROLE_YOY_DELTA]

    # Detect Periods column
    for col in df.columns:
        col_lo = col.strip().lower()
        if col_lo == "periods" or col_lo == "period":
            meta.has_periods_col = True
            meta.periods_col = col
            break
    if not meta.has_periods_col:
        # Fallback: look for a low-cardinality string column with 2 distinct values
        # that both contain 'letzte' (German for 'last') — NIQ period label pattern
        for col in df.select_dtypes(include="object").columns:
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
        period_vals = df[meta.periods_col].dropna().unique().tolist()
        for v in period_vals:
            vs = str(v)
            if " vj " in vs.lower() or vs.lower().endswith(" vj"):
                meta.py_period_label = vs
            else:
                meta.cy_period_label = vs

    # Metadata row detection: rows where dimension columns hold banner/header values
    if meta.dimension_cols:
        def _is_meta_row(row: pd.Series) -> bool:
            vals = [str(row[c]).strip().lower() for c in meta.dimension_cols
                    if c in row.index and pd.notna(row[c])]
            if not vals:
                return True
            return all(v in _METADATA_VALUES or v == "nan" for v in vals)

        meta.metadata_row_mask = df.apply(_is_meta_row, axis=1).tolist()
    else:
        meta.metadata_row_mask = [False] * len(df)

    return meta


def classify_niq_columns(df: pd.DataFrame) -> dict[str, NiqColInfo]:
    """
    Classify every column in *df* into a NIQ semantic role.

    Priority order:
      1. YoY delta  — column name (lower) contains any _YOY_PATTERNS token
      2. Prior year — column name (lower) ends with any _VJ_SUFFIXES token
      3. Numeric    — pandas dtype is numeric  → metric_cy
      4. Otherwise                              → dimension

    Also seeds descriptions and aliases from schema_lookup.SCHEMA_CATALOG
    where a matching column name exists (exact match, case-insensitive).
    """
    # Lazy import to avoid circular deps and allow the catalog to be absent
    try:
        from agent.schema_lookup import SCHEMA_CATALOG as _CATALOG
    except ImportError:
        _CATALOG = {}

    catalog_lower: dict[str, dict] = {k.lower(): v for k, v in _CATALOG.items()}

    result: dict[str, NiqColInfo] = {}

    for col in df.columns:
        col_lo = col.strip().lower()

        # --- role classification -------------------------------------------
        if any(pat in col_lo for pat in _YOY_PATTERNS):
            role = ROLE_YOY_DELTA
        elif any(col_lo.endswith(suf) for suf in _VJ_SUFFIXES):
            role = ROLE_METRIC_PY
        elif pd.api.types.is_numeric_dtype(df[col]):
            role = ROLE_METRIC_CY
        else:
            role = ROLE_DIMENSION

        # --- description & aliases from static catalog ----------------------
        cat_entry   = catalog_lower.get(col_lo, {})
        description = cat_entry.get("description", "")
        aliases     = list(cat_entry.get("aliases", []))
        if not description and cat_entry.get("en"):
            description = cat_entry["en"]

        result[col] = NiqColInfo(role=role, description=description, aliases=aliases)

    return result


def build_niq_table_context(
    meta: NiqMeta,
    dataset_name: str,
    source_filename: str,
) -> tuple[str, str, list[str]]:
    """
    Build (summary, grain, tags) for _table_context from NiqMeta.

    Returns:
        summary : one-line natural-language description
        grain   : row-level grain description
        tags    : list of keyword tags for table selection
    """
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
    if "products" in [c.lower() for c in meta.dimension_cols]:
        tags.append("product")
    if "retailers" in [c.lower() for c in meta.dimension_cols]:
        tags.extend(["retailer", "channel", "shop"])
    if "geographies" in [c.lower() for c in meta.dimension_cols]:
        tags.extend(["geography", "market", "region"])

    return summary, grain, tags


# ---------------------------------------------------------------------------
# Semantic map seeding
# ---------------------------------------------------------------------------

def seed_niq_semantic_map(
    conn,
    dataset_name: str,
    col_info: dict[str, NiqColInfo],
) -> int:
    """
    Write alias rows into _semantic_map for every column that has aliases.
    Also seeds the canonical column name itself as a searchable term.
    Returns the number of rows inserted/updated.
    """
    rows_written = 0
    for col_name, info in col_info.items():
        for alias in info.aliases:
            alias_clean = alias.strip().lower()
            if not alias_clean:
                continue
            try:
                conn.execute(
                    """
                    INSERT INTO _semantic_map (term, dataset_name, column_name, description)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (term, dataset_name)
                    DO UPDATE SET column_name = excluded.column_name,
                                  description = excluded.description
                    """,
                    [alias_clean, dataset_name, col_name, info.description or None],
                )
                rows_written += 1
            except Exception:
                pass
        col_term = col_name.strip().lower()
        if col_term:
            try:
                conn.execute(
                    """
                    INSERT INTO _semantic_map (term, dataset_name, column_name, description)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (term, dataset_name)
                    DO UPDATE SET column_name = excluded.column_name,
                                  description = excluded.description
                    """,
                    [col_term, dataset_name, col_name, info.description or None],
                )
                rows_written += 1
            except Exception:
                pass
    return rows_written


# ---------------------------------------------------------------------------
# _column_catalog enrichment
# ---------------------------------------------------------------------------

def _enrich_column_catalog(
    conn,
    dataset_name: str,
    col_info: dict[str, NiqColInfo],
) -> None:
    """Patch _column_catalog rows with NIQ-specific role flags and descriptions."""
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
# Internal helpers — imported directly from the flat tools.py module to
# avoid the circular agent.tools package → niq_ingest → agent.tools import.
# ---------------------------------------------------------------------------

def _get_flat_attr(name: str):
    """Retrieve *name* from the flat agent/tools.py module (alias _agent_tools_flat)."""
    import sys
    import importlib.util
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

    Extends the standard load_file tool with NIQ-specific post-processing:
      - Auto-detects the Periods column and CY/PY period labels
      - Classifies every column into: dimension, metric_cy, metric_py, yoy_delta
      - Drops header/banner metadata rows that NIQ exports embed in the data
      - Writes column roles and catalog descriptions to _column_catalog
      - Populates _semantic_map with natural-language aliases for every column
      - Sets grain = 'annual, CY vs PY' in _table_context

    Args:
        path: Absolute or relative path to the NIQ source file.
        dataset_name: Name to register the table / view as in DuckDB.
    """
    from agent.database import get_connection, index_table_schema
    from agent.db.catalog import upsert_table_context

    # Resolve helpers from the flat tools.py without going through the package.
    _get_parquet_store = _get_flat_attr("_get_parquet_store")
    _PROJECT_ROOT      = _get_flat_attr("_PROJECT_ROOT")

    # ── 1. Resolve file path ───────────────────────────────────────────────
    file_path = Path(path)
    if not file_path.exists():
        file_path = _PROJECT_ROOT / path
    if not file_path.exists():
        return f"ERROR: File not found at '{path}'"
    file_path = file_path.resolve()

    suffix = file_path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        elif suffix == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"ERROR: Unsupported file type '{suffix}'"
    except Exception as exc:
        return f"ERROR reading file: {exc}"

    # ── 2. Detect NIQ structure ────────────────────────────────────────────
    meta     = detect_niq_structure(df)
    col_info = classify_niq_columns(df)

    # ── 3. Drop metadata / banner rows ────────────────────────────────────
    n_raw = len(df)
    if any(meta.metadata_row_mask):
        df = df[[not m for m in meta.metadata_row_mask]].reset_index(drop=True)
    n_clean = len(df)
    dropped = n_raw - n_clean

    # ── 4. Coerce numeric columns that came in as strings (NIQ quirk) ─────
    for col, info in col_info.items():
        if info.role in (ROLE_METRIC_CY, ROLE_METRIC_PY, ROLE_YOY_DELTA):
            if col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )

    # ── 5. Write Parquet + register DuckDB view ────────────────────────────
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

    # ── 6. Update _data_registry ──────────────────────────────────────────
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

    # ── 7. Index + enrich column catalog ──────────────────────────────────
    try:
        index_table_schema(conn, dataset_name)
    except Exception as exc:
        return f"File loaded but column catalog indexing failed: {exc}"
    _enrich_column_catalog(conn, dataset_name, col_info)

    # ── 8. Seed _semantic_map ─────────────────────────────────────────────
    n_aliases = seed_niq_semantic_map(conn, dataset_name, col_info)

    # ── 9. Write _table_context ───────────────────────────────────────────
    summary, grain, tags = build_niq_table_context(meta, dataset_name, file_path.name)
    upsert_table_context(dataset_name, summary, grain, tags, conn=conn)

    # ── 10. Build response ────────────────────────────────────────────────
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

    return (
        f"Successfully loaded NIQ file '{file_path.name}' as '{dataset_name}'. "
        f"{len(df.columns)} columns, {n_clean} data rows."
        f"{drop_msg}"
        f" Column roles: {role_summary}."
        f"{periods_msg}"
        f" Seeded {n_aliases} semantic alias(es)."
    )
