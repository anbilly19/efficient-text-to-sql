"""Deterministic tool functions – no LLM involved here."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from agent.database import get_connection, get_parquet_path, PARQUET_STORE


_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _looks_like_date(value: str) -> bool:
    return bool(_DATE_PATTERN.match(str(value).strip()))


def _table_info(dataset_name: str) -> list[tuple[str, str]]:
    """Return [(column_name, column_type), ...] using PRAGMA — always safe."""
    conn = get_connection()
    rows = conn.execute(f"PRAGMA table_info('{dataset_name}')").fetchall()
    return [(r[1], r[2]) for r in rows]


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


def _normalize_date_columns(dataset_name: str) -> list[str]:
    """Cast VARCHAR date columns to DATE in-place. Returns converted column names."""
    conn = get_connection()
    converted: list[str] = []
    for col, dtype in _table_info(dataset_name):
        if dtype.upper() != "VARCHAR":
            continue
        try:
            sample = conn.execute(
                f'SELECT "{col}" FROM "{dataset_name}" WHERE "{col}" IS NOT NULL LIMIT 5'
            ).fetchall()
            values = [row[0] for row in sample if row[0] is not None]
            if values and all(_looks_like_date(v) for v in values):
                conn.execute(
                    f'ALTER TABLE "{dataset_name}" ALTER COLUMN "{col}" TYPE DATE '
                    f'USING TRY_CAST("{col}" AS DATE)'
                )
                conn.execute(
                    "UPDATE _schema_catalog SET data_type = 'DATE' "
                    "WHERE dataset_name = ? AND column_name = ?",
                    [dataset_name, col],
                )
                converted.append(col)
        except Exception:
            pass
    return converted


def _build_semantic_lookup(df: pd.DataFrame, dataset_name: str) -> None:
    """Populate _semantic_lookup with per-column stats derived from the DataFrame."""
    conn = get_connection()
    conn.execute("DELETE FROM _semantic_lookup WHERE dataset_name = ?", [dataset_name])
    total_rows = len(df)
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)

        try:
            n_distinct = int(series.nunique(dropna=True))
        except Exception:
            n_distinct = -1

        null_frac = float(series.isna().sum()) / total_rows if total_rows > 0 else 0.0

        try:
            raw_samples = series.dropna().unique()[:5].tolist()
            sample_values = json.dumps([str(s) for s in raw_samples], default=str)
        except Exception:
            sample_values = "[]"

        col_lower = col.lower().replace("_", " ")
        description = (
            f"{col_lower.title()} — {dtype_str} column with {n_distinct} distinct values"
            + (f", e.g. {', '.join(json.loads(sample_values)[:3])}" if json.loads(sample_values) else "")
        )

        conn.execute(
            """
            INSERT INTO _semantic_lookup
                (dataset_name, column_name, data_type, sample_values, n_distinct, null_frac, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [dataset_name, col, dtype_str, sample_values, n_distinct, null_frac, description],
        )


def _semantic_map_coverage(dataset_name: str, columns: list[str]) -> int:
    """Return the number of columns that have at least one alias in _semantic_map.

    The check is intentionally broad: a column is considered 'covered' if its
    exact name (case-insensitive) appears as a term in _semantic_map.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT COUNT(DISTINCT term) FROM _semantic_map "
            "WHERE LOWER(term) IN ({})".format(
                ", ".join(["LOWER(?)"] * len(columns))
            ),
            [c.lower() for c in columns],
        ).fetchone()
        return rows[0] if rows else 0
    except Exception:
        return 0


def _build_alias_csv_template(dataset_name: str, columns: list[str]) -> str:
    """Return a CSV string as an alias registration template.

    The user fills in 'definition_sql' (a SQL expression or column ref) and
    an optional 'description' then passes the file back via register_alias.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["term", "definition_sql", "description"],
        lineterminator="\n",
    )
    writer.writeheader()
    for col in columns:
        writer.writerow(
            {
                "term": col,
                "definition_sql": f'"{col}"',  # default: identity reference
                "description": "",
            }
        )
    return buf.getvalue()


@tool
def get_schema(dataset: str, columns: Optional[list[str]] = None) -> str:
    """Return column names, types, and 3 sample values for a dataset.

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
        term: The business term (e.g. 'active_customer').
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT definition_sql, description FROM _semantic_map WHERE LOWER(term) = LOWER(?)",
            [term],
        ).fetchone()
        if row:
            return json.dumps({"term": term, "definition_sql": row[0], "description": row[1]})
        return json.dumps({"term": term, "definition_sql": None, "description": "Term not found."})
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def search_semantic_lookup(query: str, dataset: Optional[str] = None) -> str:
    """Search the semantic_lookup for columns matching a keyword or description.

    Useful for finding which column represents a business concept (e.g. 'revenue',
    'customer id', 'date of order') without knowing the exact column name.

    Args:
        query: Keyword or phrase to search for in column names and descriptions.
        dataset: Optional dataset name to restrict the search to one table.
    """
    conn = get_connection()
    try:
        like_term = f"%{query.lower()}%"
        if dataset:
            rows = conn.execute(
                """
                SELECT dataset_name, column_name, data_type, sample_values, n_distinct, description
                FROM _semantic_lookup
                WHERE dataset_name = ?
                  AND (LOWER(column_name) LIKE ? OR LOWER(description) LIKE ?)
                ORDER BY dataset_name, column_name
                """,
                [dataset, like_term, like_term],
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT dataset_name, column_name, data_type, sample_values, n_distinct, description
                FROM _semantic_lookup
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
                "n_distinct": r[4],
                "description": r[5],
            }
            for r in rows
        ]
        return json.dumps(results, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def list_loaded_tables() -> str:
    """List all datasets currently registered in DuckDB.

    Returns dataset name, row count, column count, source file, and when it
    was loaded.  Falls back to scanning _semantic_lookup when _data_registry
    is empty (e.g. datasets loaded in a prior session without the registry).
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT dataset_name, source_file, row_count, column_count, loaded_at
            FROM _data_registry
            ORDER BY loaded_at DESC
            """
        ).fetchall()
        if rows:
            results = [
                {
                    "dataset": r[0],
                    "source_file": r[1],
                    "row_count": r[2],
                    "column_count": r[3],
                    "loaded_at": str(r[4]),
                }
                for r in rows
            ]
            return json.dumps(results, default=str)
    except Exception:
        pass

    # Fallback: derive from _semantic_lookup
    try:
        rows = conn.execute(
            """
            SELECT dataset_name, COUNT(*) AS column_count
            FROM _semantic_lookup
            GROUP BY dataset_name
            ORDER BY dataset_name
            """
        ).fetchall()
        results = [
            {"dataset": r[0], "column_count": r[1], "source_file": None, "row_count": None}
            for r in rows
        ]
        return json.dumps(results, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def register_alias(
    term: str,
    definition_sql: str,
    description: Optional[str] = None,
) -> str:
    """Register or update a business-term alias in the semantic map.

    After registration the alias is immediately available to lookup_semantic
    and the sql_writer uses it when constructing queries.

    Args:
        term: Short business name for the concept, e.g. 'umsatz' or 'revenue'.
        definition_sql: SQL expression that resolves the term, e.g.
            '"Umsatz_EUR"' or 'SUM("Umsatz_EUR") / 1000'.
        description: Optional human-readable explanation shown to the LLM.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO _semantic_map (term, definition_sql, description)
            VALUES (?, ?, ?)
            ON CONFLICT (term) DO UPDATE
                SET definition_sql = excluded.definition_sql,
                    description    = excluded.description
            """,
            [term.lower().strip(), definition_sql, description],
        )
        return json.dumps(
            {
                "status": "ok",
                "term": term.lower().strip(),
                "definition_sql": definition_sql,
                "description": description,
            }
        )
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def load_file(path: str, dataset_name: str, force_schema: bool = False) -> str:
    """Load an Excel, Parquet, or CSV file into DuckDB.

    Excel and CSV files are converted to Parquet and persisted on disk so they
    survive server restarts. DuckDB holds a VIEW backed by the Parquet file,
    keeping memory usage minimal. _schema_catalog, _semantic_lookup, and
    _data_registry are rebuilt for the dataset.

    If no alias entries exist in _semantic_map for this dataset's columns a
    CSV template is generated and its path is included in the response so the
    user can fill in business-term mappings and run register_alias.

    Args:
        path: Absolute or relative path to the source file.
        dataset_name: Name to register the table / view as in DuckDB.
        force_schema: If True, re-populate catalog even if entry exists.
    """
    conn = get_connection()
    file_path = Path(path)
    if not file_path.exists():
        return f"ERROR: File not found at '{path}'"

    suffix = file_path.suffix.lower()
    parquet_path = get_parquet_path(dataset_name)

    # ── 1. Read source into DataFrame ────────────────────────────────────
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
            df = _normalize_date_columns_df(df)
        elif suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
            df = _normalize_date_columns_df(df)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            return f"ERROR: Unsupported file type '{suffix}'. Use .xlsx, .parquet, or .csv."
    except Exception as exc:
        return f"ERROR reading file: {exc}"

    columns = list(df.columns)
    row_count = len(df)

    # ── 2. Persist as Parquet ─────────────────────────────────────────────
    try:
        PARQUET_STORE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
    except Exception as exc:
        return f"ERROR writing Parquet: {exc}"

    # ── 3. Register as a DuckDB VIEW backed by the Parquet file ──────────
    try:
        conn.execute(
            f"CREATE OR REPLACE VIEW \"{dataset_name}\" AS "
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        )
    except Exception as exc:
        return f"ERROR registering view in DuckDB: {exc}"

    # ── 4. Rebuild _schema_catalog ────────────────────────────────────────
    try:
        col_info = _table_info(dataset_name)
        conn.execute("DELETE FROM _schema_catalog WHERE dataset_name = ?", [dataset_name])
        for col, dtype in col_info:
            conn.execute(
                "INSERT INTO _schema_catalog "
                "(dataset_name, column_name, data_type, nullable, description) "
                "VALUES (?, ?, ?, ?, ?)",
                [dataset_name, col, dtype, True, None],
            )
    except Exception as exc:
        return f"File loaded but schema catalog update failed: {exc}"

    # ── 5. Build _semantic_lookup ─────────────────────────────────────────
    try:
        _build_semantic_lookup(df, dataset_name)
    except Exception:
        pass  # non-fatal

    # ── 6. Update _data_registry ──────────────────────────────────────────
    try:
        conn.execute(
            """
            INSERT INTO _data_registry
                (dataset_name, parquet_path, source_file, row_count, column_count, loaded_at)
            VALUES (?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (dataset_name) DO UPDATE
                SET parquet_path  = excluded.parquet_path,
                    source_file   = excluded.source_file,
                    row_count     = excluded.row_count,
                    column_count  = excluded.column_count,
                    loaded_at     = excluded.loaded_at
            """,
            [
                dataset_name,
                parquet_path.as_posix(),
                file_path.name,
                row_count,
                len(columns),
            ],
        )
    except Exception:
        pass  # non-fatal

    # ── 7. Check semantic_map coverage → emit template if missing ─────────
    alias_warning = ""
    template_path: Optional[Path] = None
    coverage = _semantic_map_coverage(dataset_name, columns)
    if coverage == 0:
        try:
            template_csv = _build_alias_csv_template(dataset_name, columns)
            template_dir = PARQUET_STORE.parent / "templates"
            template_dir.mkdir(parents=True, exist_ok=True)
            template_path = template_dir / f"{dataset_name}_aliases.csv"
            template_path.write_text(template_csv, encoding="utf-8")
            alias_warning = (
                f" No business-term aliases found for '{dataset_name}'. "
                f"A template CSV has been saved to '{template_path}'. "
                f"Fill in 'definition_sql' and 'description' for each column, "
                f"then use register_alias to load them."
            )
        except Exception:
            alias_warning = (
                f" No business-term aliases found for '{dataset_name}'. "
                f"Use register_alias(term, definition_sql) to map column names to business terms."
            )

    return (
        f"Successfully loaded '{file_path.name}' as view '{dataset_name}'. "
        f"{len(col_info)} columns, {row_count:,} rows. "
        f"Persisted to '{parquet_path}'."
        + alias_warning
    )


ALL_TOOLS = [
    get_schema,
    profile_column,
    run_sql,
    run_test_query,
    lookup_semantic,
    search_semantic_lookup,
    list_loaded_tables,
    register_alias,
    load_file,
]
ORCHESTRATOR_TOOLS = [get_schema, lookup_semantic, list_loaded_tables]
PROFILER_TOOLS = [get_schema, profile_column]
SQL_WRITER_TOOLS = [get_schema, lookup_semantic, search_semantic_lookup, profile_column]
VERIFIER_TOOLS = [run_test_query, profile_column]
