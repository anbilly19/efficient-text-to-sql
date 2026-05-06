"""Deterministic tool functions – no LLM involved here."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from agent.database import get_connection


_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _looks_like_date(value: str) -> bool:
    return bool(_DATE_PATTERN.match(str(value).strip()))


def _table_info(dataset_name: str) -> list[tuple[str, str]]:
    """Return [(column_name, column_type), ...] using PRAGMA — always safe."""
    conn = get_connection()
    rows = conn.execute(f"PRAGMA table_info('{dataset_name}')").fetchall()
    # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
    return [(r[1], r[2]) for r in rows]


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
def load_file(path: str, dataset_name: str, force_schema: bool = False) -> str:
    """Load an Excel, Parquet, or CSV file into DuckDB and register its schema.

    Args:
        path: Absolute or relative path to the file.
        dataset_name: Name to register the table as in DuckDB.
        force_schema: If True, re-populate _schema_catalog even if entry exists.
    """
    conn = get_connection()
    file_path = Path(path)
    if not file_path.exists():
        return f"ERROR: File not found at '{path}'"

    suffix = file_path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
            conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
            conn.register("_tmp_load", df)
            conn.execute(f'CREATE TABLE "{dataset_name}" AS SELECT * FROM _tmp_load')
            conn.unregister("_tmp_load")
        elif suffix == ".parquet":
            conn.execute(
                f'CREATE OR REPLACE TABLE "{dataset_name}" AS SELECT * FROM read_parquet(\'{path}\')'
            )
        elif suffix == ".csv":
            conn.execute(
                f'CREATE OR REPLACE TABLE "{dataset_name}" AS SELECT * FROM read_csv_auto(\'{path}\')'
            )
        else:
            return f"ERROR: Unsupported file type '{suffix}'. Use .xlsx, .parquet, or .csv."
    except Exception as exc:
        return f"ERROR loading file: {exc}"

    converted = _normalize_date_columns(dataset_name)

    try:
        col_info = _table_info(dataset_name)
        conn.execute("DELETE FROM _schema_catalog WHERE dataset_name = ?", [dataset_name])
        for col, dtype in col_info:
            conn.execute(
                "INSERT INTO _schema_catalog (dataset_name, column_name, data_type, nullable, description) "
                "VALUES (?, ?, ?, ?, ?)",
                [dataset_name, col, dtype, True, None],
            )
    except Exception as exc:
        return f"File loaded as '{dataset_name}' but schema catalog update failed: {exc}"

    row_count = conn.execute(f'SELECT COUNT(*) FROM "{dataset_name}"').fetchone()[0]
    converted_note = f" Auto-converted date columns: {converted}." if converted else ""
    return (
        f"Successfully loaded '{path}' as table '{dataset_name}'. "
        f"{len(col_info)} columns, {row_count} rows registered.{converted_note}"
    )


ALL_TOOLS = [get_schema, profile_column, run_sql, run_test_query, lookup_semantic, load_file]
ORCHESTRATOR_TOOLS = [get_schema, lookup_semantic]
PROFILER_TOOLS = [get_schema, profile_column]
SQL_WRITER_TOOLS = [get_schema, lookup_semantic, profile_column]
VERIFIER_TOOLS = [run_test_query, profile_column]
