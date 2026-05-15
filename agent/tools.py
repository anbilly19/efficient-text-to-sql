"""Deterministic tool functions – no LLM involved here."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Lazy database accessors
# Do NOT do `from agent.database import get_connection` at module level.
# That binds to the module object that existed at import time. If tests
# purge and re-import agent.database, the bound name becomes stale and
# load_file ends up writing views onto a different connection than the one
# the caller later queries.
#
# Instead, always go through the live sys.modules entry at call-time.
# ---------------------------------------------------------------------------

import sys as _sys


def _db():
    """Return the live agent.database module."""
    return _sys.modules["agent.database"]


def get_connection():
    return _db().get_connection()


def get_schema_context(table_names):
    return _db().get_schema_context(table_names)


def get_table_summaries():
    return _db().get_table_summaries()


def index_table_schema(conn, dataset_name):
    return _db().index_table_schema(conn, dataset_name)


def infer_and_register_relationships(conn, new_table):
    return _db().infer_and_register_relationships(conn, new_table)


def register_relationship(conn, left_table, left_column, right_table, right_column,
                          cardinality="many-to-one", description=None):
    return _db().register_relationship(
        conn, left_table, left_column, right_table, right_column, cardinality, description
    )


# Ensure agent.database is imported (it always will be, but be explicit).
import agent.database as _agent_database  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Parquet storage
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PARQUET_STORE = _PROJECT_ROOT / ".local" / "parquet"


def _get_parquet_store() -> Path:
    """Return the active PARQUET_STORE path, reading the env var at call-time."""
    env = os.environ.get("PARQUET_STORE")
    return Path(env) if env else _DEFAULT_PARQUET_STORE


def get_parquet_path(dataset_name: str) -> Path:
    """Return the absolute path where *dataset_name* is (or will be) stored."""
    return _get_parquet_store() / f"{dataset_name}.parquet"


class _LazyParquetStore:
    def __truediv__(self, other):   return _get_parquet_store() / other
    def __str__(self):              return str(_get_parquet_store())
    def __repr__(self):             return repr(_get_parquet_store())
    def __fspath__(self):           return os.fspath(_get_parquet_store())
    def mkdir(self, **kw):          return _get_parquet_store().mkdir(**kw)
    def exists(self):               return _get_parquet_store().exists()
    def __eq__(self, other):        return _get_parquet_store() == other


PARQUET_STORE = _LazyParquetStore()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _looks_like_date(value: str) -> bool:
    return bool(_DATE_PATTERN.match(str(value).strip()))


def _normalize_date_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns that look like ISO dates to datetime64."""
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(5)
        if len(sample) > 0 and all(_looks_like_date(str(v)) for v in sample):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def _table_info(dataset_name: str) -> list[tuple[str, str]]:
    """Return [(column_name, column_type), ...] via information_schema."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [dataset_name],
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


# ---------------------------------------------------------------------------
# NIQ-specific heuristics
# ---------------------------------------------------------------------------

# Keywords that strongly indicate a metric column in NIQ panel data
_NIQ_METRIC_KEYWORDS = {
    "penetration", "spend", "buyer", "volume", "sales",
    "trips", "units", "frequency", "share", "revenue",
    "amount", "value", "rate", "index", "count",
}

# Keywords that indicate a dimension / label column regardless of dtype
_NIQ_DIMENSION_KEYWORDS = {
    "period", "product", "category", "brand", "market",
    "channel", "retailer", "region", "segment", "pack",
    "manufacturer", "supplier", "description", "name", "label",
}

# Keywords that tag a metric as prior-year / YoY comparison
_NIQ_PY_KEYWORDS = {"py", "prior year", "prev", "previous", "yoy", "year on year", "year-on-year"}

# Keywords that tag a metric as current-year
_NIQ_CY_KEYWORDS = {"cy", "current year", "curr", "current"}


def classify_niq_columns(conn, dataset_name: str) -> None:
    """Patch _column_catalog with NIQ-aware roles and descriptions.

    Rules (in priority order):
    1. If the column name contains a dimension keyword → force is_dimension=True,
       is_metric=False, regardless of dtype.
    2. If the column name contains a metric keyword → is_metric=True,
       is_dimension=False, with PY/CY sub-tagging in description.
    3. Otherwise fall back to the dtype-based classification already written
       by index_table_schema (no change).
    """
    cols = conn.execute(
        "SELECT column_name, column_type, is_metric, is_dimension FROM _column_catalog "
        "WHERE dataset_name = ?",
        [dataset_name],
    ).fetchall()

    for col_name, col_type, cur_metric, cur_dim in cols:
        lower = col_name.lower()
        desc: Optional[str] = None
        is_metric: Optional[bool] = None
        is_dim: Optional[bool] = None

        if any(k in lower for k in _NIQ_DIMENSION_KEYWORDS):
            is_metric, is_dim = False, True
            desc = "NIQ dimension"
        elif any(k in lower for k in _NIQ_METRIC_KEYWORDS):
            is_metric, is_dim = True, False
            if any(k in lower for k in _NIQ_PY_KEYWORDS):
                desc = "NIQ metric — prior year / YoY comparison"
            elif any(k in lower for k in _NIQ_CY_KEYWORDS):
                desc = "NIQ metric — current year"
            else:
                desc = "NIQ metric"
        else:
            # No keyword match — keep existing dtype-derived flags, only set desc if numeric
            is_metric = cur_metric
            is_dim = cur_dim
            desc = "NIQ metric" if cur_metric else "NIQ dimension"

        conn.execute(
            "UPDATE _column_catalog "
            "SET is_metric = ?, is_dimension = ?, description = ? "
            "WHERE dataset_name = ? AND column_name = ?",
            [is_metric, is_dim, desc, dataset_name, col_name],
        )


def detect_niq_structure(conn, dataset_name: str) -> dict:
    """Detect grain and period labels by inspecting actual column values.

    Returns a dict with at least:
      grain        – "annual", "annual, CY vs PY", or "unknown"
      period_col   – name of the period column found, or None
      period_values – sample distinct values from that column
    """
    try:
        period_row = conn.execute(
            "SELECT column_name FROM _column_catalog "
            "WHERE dataset_name = ? AND LOWER(column_name) LIKE '%period%' LIMIT 1",
            [dataset_name],
        ).fetchone()

        grain = "annual"
        period_col = period_row[0] if period_row else None
        period_values: list = []

        if period_col:
            rows = conn.execute(
                f'SELECT DISTINCT "{period_col}" FROM "{dataset_name}" '
                f'WHERE "{period_col}" IS NOT NULL LIMIT 20'
            ).fetchall()
            period_values = [str(r[0]) for r in rows]
            lowered = [v.lower() for v in period_values]
            has_cy = any("cy" in v or "current" in v for v in lowered)
            has_py = any("py" in v or "prior" in v or "yoy" in v for v in lowered)
            if has_cy or has_py:
                grain = "annual, CY vs PY"

        return {
            "grain": grain,
            "period_col": period_col,
            "period_values": period_values,
        }
    except Exception as exc:
        return {"grain": "unknown", "period_col": None, "period_values": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# NIQ semantic map
#
# IMPORTANT: column_name values here must exactly match the real DuckDB column
# names in the loaded NIQ file (as seen in _column_catalog / information_schema).
#
# Confirmed real column names from NIQ-Haushaltspaneldaten_synthetic.xlsx:
#   Penetration (%)                        → penetration metric CY
#   Penetration (%) VJ                     → penetration metric PY
#   Penetration (%) vs. VJ (% Ver.)        → YoY delta for penetration
#   Ausgaben pro Käuferhaushalt            → spend per buyer CY
#   Ausgaben pro Käuferhaushalt VJ         → spend per buyer PY
#   Ausgaben pro Käuferhaushalt vs. VJ ... → YoY delta for spend per buyer
#   Käuferhaushalte                        → buying households (absolute count)
#   Käuferhaushalte VJ                     → buying households PY
# ---------------------------------------------------------------------------

# Each entry: (term, column_name, description)
# dataset_name is filled in at load time so aliases are scoped per dataset.
_NIQ_SEMANTIC_TERMS: list[tuple[str, str, str]] = [
    # --- Penetration / Käuferreichweite ---
    ("penetration",              "Penetration (%)",                    "% of households buying at least once (CY)"),
    ("Käuferreichweite",         "Penetration (%)",                    "German: buyer reach → Penetration (%)"),
    ("käuferreichweite",         "Penetration (%)",                    "German (lowercase): buyer reach → Penetration (%)"),
    ("buyer reach",              "Penetration (%)",                    "Alias for Penetration (%)"),
    ("buyer penetration",        "Penetration (%)",                    "Alias for Penetration (%)"),
    ("Haushaltsdurchdringung",   "Penetration (%)",                    "German: household penetration → Penetration (%)"),
    ("haushaltsdurchdringung",   "Penetration (%)",                    "German (lowercase): household penetration"),

    # --- Spend per buyer / Ausgaben pro Käuferhaushalt ---
    ("spend per buyer",          "Ausgaben pro Käuferhaushalt",        "Average spend in € per buying household (CY)"),
    ("Ausgaben je Käufer",       "Ausgaben pro Käuferhaushalt",        "German: spend per buyer → Ausgaben pro Käuferhaushalt"),
    ("ausgaben je käufer",       "Ausgaben pro Käuferhaushalt",        "German (lowercase): spend per buyer"),
    ("Ausgaben pro Käufer",      "Ausgaben pro Käuferhaushalt",        "German variant: spend per buyer"),
    ("ausgaben pro käufer",      "Ausgaben pro Käuferhaushalt",        "German (lowercase) variant: spend per buyer"),

    # --- YoY / Veränderung ---
    ("yoy",                      "Penetration (%) vs. VJ (% Ver.)",   "Year-over-year change (penetration default)"),
    ("year on year",             "Penetration (%) vs. VJ (% Ver.)",   "Year-over-year change vs prior year"),
    ("year-on-year",             "Penetration (%) vs. VJ (% Ver.)",   "Year-over-year change vs prior year"),
    ("Veränderung zum Vorjahr",  "Penetration (%) vs. VJ (% Ver.)",   "German: change vs prior year → Penetration (%) vs. VJ (% Ver.)"),
    ("veränderung zum vorjahr",  "Penetration (%) vs. VJ (% Ver.)",   "German (lowercase): change vs prior year"),

    # --- Buying households ---
    ("buying households",        "Käuferhaushalte",                    "Absolute count of buying households (CY)"),
    ("Käuferhaushalte",          "Käuferhaushalte",                    "German: buying households"),
]


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

@tool
def list_tables() -> str:
    """Return a short summary of every loaded table with row/column counts.

    Use this first when the user's question may span multiple tables, or when
    you need to know which tables are available before selecting the right ones.
    """
    return get_table_summaries()


@tool
def get_relationships() -> str:
    """Return all known join relationships between loaded tables.

    Returns a JSON list of objects with keys:
      left_table, left_column, right_table, right_column, cardinality, description.
    Use this before writing a JOIN to confirm which columns to join on.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT left_table, left_column, right_table, right_column, cardinality, description "
            "FROM _relationships ORDER BY left_table, right_table"
        ).fetchall()
        result = [
            {
                "left_table": r[0],
                "left_column": r[1],
                "right_table": r[2],
                "right_column": r[3],
                "cardinality": r[4],
                "description": r[5],
            }
            for r in rows
        ]
        if not result:
            return json.dumps({"relationships": [], "note": "No relationships registered yet."})
        return json.dumps({"relationships": result}, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def get_schema_context_tool(table_names: list[str]) -> str:
    """Return full schema context (columns, types, flags, samples, relationships) for the
    given list of table names.  Always call this before writing SQL that touches
    any of those tables.

    Args:
        table_names: One or more table names to describe.
    """
    try:
        return get_schema_context(table_names)
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def register_relationship_tool(
    left_table: str,
    left_column: str,
    right_table: str,
    right_column: str,
    cardinality: str = "many-to-one",
    description: Optional[str] = None,
) -> str:
    """Explicitly register (or update) an authoritative join relationship.

    Use this when you've verified the correct join keys through data exploration
    and want to record them for future queries.

    Args:
        left_table: The fact / left-hand table name.
        left_column: The join column on the left table.
        right_table: The dimension / right-hand table name.
        right_column: The join column on the right table.
        cardinality: e.g. 'many-to-one', 'one-to-one'. Default 'many-to-one'.
        description: Optional human-readable note.
    """
    try:
        register_relationship(
            get_connection(),
            left_table,
            left_column,
            right_table,
            right_column,
            cardinality,
            description,
        )
        return f"Registered: {left_table}.{left_column} → {right_table}.{right_column} ({cardinality})"
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def get_schema(dataset: str, columns: Optional[list[str]] = None) -> str:
    """Return column names, types, and 3 sample values for a single dataset.

    Args:
        dataset: The table / view name in DuckDB.
        columns: Optional subset of column names to describe.
    """
    conn = get_connection()
    try:
        col_info = _table_info(dataset)
    except Exception as exc:
        return f"ERROR: {exc}"

    if columns:
        col_info = [(c, t) for c, t in col_info if c in columns]

    result: list[dict] = []
    for col, dtype in col_info:
        try:
            samples = conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{dataset}" WHERE "{col}" IS NOT NULL LIMIT 3'
            ).fetchall()
            samples = [str(s[0]) for s in samples]
        except Exception:
            samples = []
        result.append({"column": col, "type": dtype, "samples": samples})

    return json.dumps(result, default=str)


@tool
def profile_column(dataset: str, column: str) -> str:
    """Return null%, top 10 distinct values, and numeric stats for a column.

    Args:
        dataset: Table name in DuckDB.
        column: Column to profile.
    """
    conn = get_connection()
    try:
        total = conn.execute(f'SELECT COUNT(*) FROM "{dataset}"').fetchone()[0]
        nulls = conn.execute(
            f'SELECT COUNT(*) FROM "{dataset}" WHERE "{column}" IS NULL'
        ).fetchone()[0]
        null_pct = round(100 * nulls / total, 2) if total else 0

        top_rows = conn.execute(
            f'SELECT "{column}", COUNT(*) AS cnt FROM "{dataset}" '
            f'GROUP BY "{column}" ORDER BY cnt DESC LIMIT 10'
        ).fetchall()
        top_vals = [{column: str(r[0]), "cnt": r[1]} for r in top_rows]

        stats: dict = {"null_pct": null_pct, "top_values": top_vals}

        try:
            num_stats = conn.execute(
                f'SELECT MIN("{column}"), MAX("{column}"), AVG("{column}") FROM "{dataset}"'
            ).fetchone()
            stats["min"] = str(num_stats[0])
            stats["max"] = str(num_stats[1])
            stats["mean"] = str(num_stats[2])
        except Exception:
            pass

        return json.dumps(stats, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


def _execute_select(query: str) -> tuple[str, dict]:
    """Run a read-only SELECT; return (rows_json, metadata). Raises on forbidden DDL."""
    conn = get_connection()
    normalised = query.strip().upper()
    for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "COPY"):
        if re.search(rf"\b{forbidden}\b", normalised):
            raise ValueError(f"Forbidden keyword '{forbidden}' in query.")

    rel = conn.execute(query)
    rows = rel.fetchall()
    columns = [desc[0] for desc in rel.description]

    records = [dict(zip(columns, row)) for row in rows]
    limited = records[:50]
    rows_json = json.dumps(limited, default=str)

    df_for_stats = pd.DataFrame(records)
    metadata: dict = {
        "row_count": len(records),
        "returned_rows": len(limited),
        "columns": columns,
    }
    checksums: dict = {}
    for col in df_for_stats.select_dtypes(include="number").columns:
        checksums[col] = {
            "sum": float(df_for_stats[col].sum()),
            "mean": float(df_for_stats[col].mean()),
        }
    metadata["checksums"] = checksums
    return rows_json, metadata


@tool
def run_sql(query: str) -> str:
    """Execute a read-only SELECT query and return the first 50 rows as JSON.

    Args:
        query: A valid DuckDB SELECT statement.
    """
    try:
        rows_json, meta = _execute_select(query)
        return json.dumps({"rows": json.loads(rows_json), "metadata": meta}, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def run_test_query(query: str) -> str:
    """Like run_sql but also returns checksums for verification.

    Args:
        query: A valid DuckDB SELECT statement.
    """
    try:
        rows_json, meta = _execute_select(query)
        return json.dumps({"rows": json.loads(rows_json), "metadata": meta}, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def lookup_semantic(term: str) -> str:
    """Look up a business term in the semantic map.

    Args:
        term: The business term (e.g. 'revenue', 'active_customer').
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT description FROM _semantic_map WHERE LOWER(term) = LOWER(?)",
            [term],
        ).fetchone()
        if row:
            cols = conn.execute(
                "SELECT dataset_name, column_name, column_type, sample_values "
                "FROM _column_catalog WHERE LOWER(column_name) LIKE LOWER(?)",
                [f"%{term}%"],
            ).fetchall()
            return json.dumps({
                "term": term,
                "description": row[0],
                "matching_columns": [
                    {"table": r[0], "column": r[1], "type": r[2], "samples": json.loads(r[3] or "[]")}
                    for r in cols
                ],
            })
        cols = conn.execute(
            "SELECT dataset_name, column_name, column_type, description, sample_values "
            "FROM _column_catalog WHERE LOWER(column_name) LIKE LOWER(?)",
            [f"%{term}%"],
        ).fetchall()
        if cols:
            return json.dumps({
                "term": term,
                "description": None,
                "matching_columns": [
                    {"table": r[0], "column": r[1], "type": r[2],
                     "desc": r[3], "samples": json.loads(r[4] or "[]")}
                    for r in cols
                ],
            })
        return json.dumps({"term": term, "description": None, "matching_columns": [],
                           "note": "Term not found in semantic map or column catalog."})
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def search_semantic_lookup(query: str, dataset: Optional[str] = None) -> str:
    """Search _column_catalog for columns matching a keyword or description.

    Args:
        query: Keyword or phrase to search for in column names and descriptions.
        dataset: Optional table name to restrict the search.
    """
    conn = get_connection()
    try:
        like_term = f"%{query.lower()}%"
        if dataset:
            rows = conn.execute(
                """
                SELECT dataset_name, column_name, column_type, sample_values,
                       is_metric, is_dimension, is_join_key, description
                FROM _column_catalog
                WHERE dataset_name = ?
                  AND (LOWER(column_name) LIKE ? OR LOWER(description) LIKE ?)
                ORDER BY dataset_name, column_name
                """,
                [dataset, like_term, like_term],
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT dataset_name, column_name, column_type, sample_values,
                       is_metric, is_dimension, is_join_key, description
                FROM _column_catalog
                WHERE LOWER(column_name) LIKE ? OR LOWER(description) LIKE ?
                ORDER BY dataset_name, column_name
                """,
                [like_term, like_term],
            ).fetchall()
        results = [
            {
                "dataset": r[0],
                "column": r[1],
                "type": r[2],
                "samples": json.loads(r[3] or "[]"),
                "is_metric": r[4],
                "is_dimension": r[5],
                "is_join_key": r[6],
                "description": r[7],
            }
            for r in rows
        ]
        return json.dumps(results, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def load_file(path: str, dataset_name: str) -> str:
    """Load an Excel, Parquet, or CSV file into DuckDB.

    Converts the file to Parquet, registers it as a DuckDB VIEW, populates
    _data_registry, _column_catalog, and _table_context, and auto-detects
    join relationships to any already-loaded tables.

    Args:
        path: Absolute or relative path to the source file.
        dataset_name: Name to register the table / view as in DuckDB.
    """
    conn = get_connection()

    file_path = Path(path)
    if not file_path.exists():
        file_path = _PROJECT_ROOT / path
    if not file_path.exists():
        return f"ERROR: File not found at '{path}' (also tried '{_PROJECT_ROOT / path}')"
    file_path = file_path.resolve()

    suffix = file_path.suffix.lower()
    parquet_store = _get_parquet_store()
    parquet_path = parquet_store / f"{dataset_name}.parquet"

    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
            df = _normalize_date_columns_df(df)
        elif suffix == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
            df = _normalize_date_columns_df(df)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"ERROR: Unsupported file type '{suffix}'. Use .xlsx, .parquet, or .csv."
    except Exception as exc:
        return f"ERROR reading file: {exc}"

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
        return f"ERROR registering view in DuckDB: {exc}"

    row_count = len(df)
    col_count = len(df.columns)

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
            [dataset_name, str(parquet_path.resolve()), str(file_path), row_count, col_count],
        )
    except Exception as exc:
        return f"File loaded but registry update failed: {exc}"

    try:
        index_table_schema(conn, dataset_name)
    except Exception as exc:
        return f"File loaded but column catalog indexing failed: {exc}"

    inferred = []
    try:
        inferred = infer_and_register_relationships(conn, dataset_name)
    except Exception:
        pass

    try:
        existing = conn.execute(
            "SELECT 1 FROM _table_context WHERE dataset_name = ?", [dataset_name]
        ).fetchone()
        if not existing:
            conn.execute(
                """
                INSERT INTO _table_context (dataset_name, summary, grain)
                VALUES (?, ?, ?)
                """,
                [
                    dataset_name,
                    f"Data loaded from {file_path.name} ({row_count} rows, {col_count} columns).",
                    None,
                ],
            )
    except Exception:
        pass

    inferred_msg = (
        f" Auto-detected {len(inferred)} relationship(s): "
        + ", ".join(f"{r['left']}.{r['on']} ↔ {r['right']}.{r['on']}" for r in inferred)
        if inferred
        else ""
    )

    return (
        f"Successfully loaded '{file_path.name}' as view '{dataset_name}'. "
        f"{col_count} columns, {row_count} rows."
        f"{inferred_msg}"
    )


@tool
def load_niq_file(path: str, dataset_name: str) -> str:
    """Load a NIQ panel Excel/CSV/Parquet file with NIQ-aware column classification.

    Extends load_file with:
    - NIQ-specific column role detection (classify_niq_columns): correctly
      tags Periods/Products/Category as dimensions and Penetration/Spend/YoY
      as metrics with PY vs CY sub-tagging.
    - Grain detection (detect_niq_structure): inspects period-column values to
      determine "annual" vs "annual, CY vs PY" and writes to _table_context.
    - Semantic map seeding: registers German/English marketing term aliases
      (Käuferreichweite, Ausgaben je Käufer, yoy, etc.) scoped to this dataset.

    The semantic map entries point to the EXACT column names present in the
    NIQ file (e.g. 'Penetration (%)' not 'Penetration %'). This ensures
    lookup_semantic and search_semantic_lookup resolve aliases without errors.

    For multi-dataset NIQ support, use distinct dataset_names per extract
    (e.g. niq_panel_petfood, niq_panel_snacks). Repeated calls upsert safely.

    Args:
        path: Absolute or relative path to the NIQ source file.
        dataset_name: Name to register the view as in DuckDB (e.g. 'niq_panel').
    """
    # Step 1: generic ingestion (Parquet conversion, registry, catalog, relationships)
    result = load_file.invoke({"path": path, "dataset_name": dataset_name})
    if result.startswith("ERROR"):
        return result

    conn = get_connection()

    # Step 2: patch _column_catalog with NIQ-aware roles and descriptions
    try:
        classify_niq_columns(conn, dataset_name)
    except Exception as exc:
        return result + f" | WARNING: classify_niq_columns failed: {exc}"

    # Step 3: detect grain from actual data, update _table_context
    structure = detect_niq_structure(conn, dataset_name)
    grain = structure.get("grain", "unknown")
    try:
        conn.execute(
            """
            UPDATE _table_context
            SET grain   = ?,
                summary = ?,
                tags    = 'niq,panel'
            WHERE dataset_name = ?
            """,
            [
                grain,
                (
                    f"NIQ panel dataset '{dataset_name}'. "
                    f"Grain: {grain}. "
                    f"Period column: {structure.get('period_col') or 'not detected'}."
                ),
                dataset_name,
            ],
        )
    except Exception as exc:
        return result + f" | WARNING: _table_context update failed: {exc}"

    # Step 4: seed _semantic_map with German/English aliases scoped to this dataset.
    # column_name values MUST match exact DuckDB column names from the NIQ file.
    niq_rows = [
        (term, dataset_name, col, desc)
        for term, col, desc in _NIQ_SEMANTIC_TERMS
    ]
    try:
        conn.executemany(
            """
            INSERT INTO _semantic_map (term, dataset_name, column_name, description)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (term, dataset_name) DO UPDATE SET
                column_name = excluded.column_name,
                description = excluded.description
            """,
            niq_rows,
        )
    except Exception as exc:
        return result + f" | WARNING: _semantic_map seeding failed: {exc}"

    period_info = (
        f"Periods: CY/PY detected in '{structure['period_col']}' → {structure['period_values'][:3]}"
        if structure.get("period_values")
        else "No period column detected"
    )
    return (
        result
        + f" | NIQ grain: {grain}"
        + f" | {period_info}"
        + f" | Seeded {len(niq_rows)} semantic alias(es)"
    )


# ---------------------------------------------------------------------------
# Tool sets per node
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    list_tables,
    get_relationships,
    get_schema_context_tool,
    register_relationship_tool,
    get_schema,
    profile_column,
    run_sql,
    run_test_query,
    lookup_semantic,
    search_semantic_lookup,
    load_file,
    load_niq_file,
]

ORCHESTRATOR_TOOLS = [
    list_tables,
    get_relationships,
    get_schema_context_tool,
    lookup_semantic,
]

PROFILER_TOOLS = [
    get_schema_context_tool,
    get_schema,
    profile_column,
]

SQL_WRITER_TOOLS = [
    get_schema_context_tool,
    get_relationships,
    lookup_semantic,
    search_semantic_lookup,
    profile_column,
]

VERIFIER_TOOLS = [
    run_test_query,
    profile_column,
    get_relationships,
]