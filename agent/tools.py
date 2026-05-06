"""Deterministic tool functions – no LLM involved here.

All tools are wrapped as LangChain @tool so they can be bound to LLM nodes.
The LLM may only produce SQL strings; all execution is done in these tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from agent.database import get_connection


# ── Schema & Catalog ───────────────────────────────────────────────────────────────────────


@tool
def get_schema(dataset: str, columns: Optional[list[str]] = None) -> str:
    """Return column names, types, and 3 random sample values for a dataset.

    Args:
        dataset: The table / view name in DuckDB.
        columns: Optional subset of column names to describe.
    """
    conn = get_connection()
    try:
        desc = conn.execute(f"DESCRIBE {dataset}").fetchdf()
    except Exception as exc:
        return f"ERROR: {exc}"

    if columns:
        desc = desc[desc["column_name"].isin(columns)]

    result: list[dict] = []
    for _, row in desc.iterrows():
        col = row["column_name"]
        dtype = row["column_type"]
        try:
            samples = conn.execute(
                f"SELECT DISTINCT {col} FROM {dataset} WHERE {col} IS NOT NULL LIMIT 3"
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
        total = conn.execute(f"SELECT COUNT(*) FROM {dataset}").fetchone()[0]
        nulls = conn.execute(
            f"SELECT COUNT(*) FROM {dataset} WHERE {column} IS NULL"
        ).fetchone()[0]
        null_pct = round(100 * nulls / total, 2) if total else 0

        top_vals = conn.execute(
            f"""
            SELECT {column}, COUNT(*) AS cnt
            FROM {dataset}
            GROUP BY {column}
            ORDER BY cnt DESC
            LIMIT 10
            """
        ).fetchdf().to_dict(orient="records")

        stats: dict = {"null_pct": null_pct, "top_values": top_vals}

        try:
            num_stats = conn.execute(
                f"SELECT MIN({column}), MAX({column}), AVG({column}) FROM {dataset}"
            ).fetchone()
            stats["min"] = str(num_stats[0])
            stats["max"] = str(num_stats[1])
            stats["mean"] = str(num_stats[2])
        except Exception:
            pass

        try:
            date_range = conn.execute(
                f"SELECT MIN({column})::DATE, MAX({column})::DATE FROM {dataset}"
            ).fetchone()
            stats["date_min"] = str(date_range[0])
            stats["date_max"] = str(date_range[1])
        except Exception:
            pass

        return json.dumps(stats, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


# ── SQL Execution ───────────────────────────────────────────────────────────────────────


def _execute_select(query: str) -> tuple[str, dict]:
    """Internal helper – runs a SELECT and returns (rows_json, metadata)."""
    conn = get_connection()
    normalised = query.strip().upper()
    for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "COPY"):
        if forbidden in normalised:
            raise ValueError(f"Forbidden keyword '{forbidden}' in query.")

    df: pd.DataFrame = conn.execute(query).fetchdf()
    limited = df.head(50)
    rows_json = limited.to_json(orient="records", default_handler=str)
    metadata = {
        "row_count": len(df),
        "returned_rows": len(limited),
        "columns": list(df.columns),
    }
    checksums: dict = {}
    for col in df.select_dtypes(include="number").columns:
        checksums[col] = {"sum": float(df[col].sum()), "mean": float(df[col].mean())}
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
    """Like run_sql but also returns row count and numeric checksums for verification.

    Args:
        query: A valid DuckDB SELECT statement.
    """
    try:
        rows_json, meta = _execute_select(query)
        return json.dumps({"rows": json.loads(rows_json), "metadata": meta}, default=str)
    except Exception as exc:
        return f"ERROR: {exc}"


# ── Semantic Layer ──────────────────────────────────────────────────────────────────────


@tool
def lookup_semantic(term: str) -> str:
    """Look up a business term in the semantic map and return its SQL definition.

    Args:
        term: The business term to look up (e.g. 'active_customer').
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT definition_sql, description FROM _semantic_map WHERE LOWER(term) = LOWER(?)",
            [term],
        ).fetchone()
        if row:
            return json.dumps({"term": term, "definition_sql": row[0], "description": row[1]})
        return json.dumps({"term": term, "definition_sql": None, "description": "Term not found in semantic map."})
    except Exception as exc:
        return f"ERROR: {exc}"


# ── Data Loading ───────────────────────────────────────────────────────────────────────


@tool
def load_file(path: str, dataset_name: str, force_schema: bool = False) -> str:
    """Load an Excel, Parquet, or CSV file into DuckDB and register its schema.

    The data is persisted as a real DuckDB table so it survives across calls.

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
            # Read into pandas then persist as a real DuckDB table
            df = pd.read_excel(path)
            conn.execute(f"DROP TABLE IF EXISTS {dataset_name}")
            conn.register("_tmp_load", df)
            conn.execute(f"CREATE TABLE {dataset_name} AS SELECT * FROM _tmp_load")
            conn.unregister("_tmp_load")
        elif suffix == ".parquet":
            conn.execute(f"CREATE OR REPLACE TABLE {dataset_name} AS SELECT * FROM read_parquet('{path}')")
        elif suffix == ".csv":
            conn.execute(f"CREATE OR REPLACE TABLE {dataset_name} AS SELECT * FROM read_csv_auto('{path}')")
        else:
            return f"ERROR: Unsupported file type '{suffix}'. Use .xlsx, .parquet, or .csv."
    except Exception as exc:
        return f"ERROR loading file: {exc}"

    try:
        desc = conn.execute(f"DESCRIBE {dataset_name}").fetchdf()
        # Clear old catalog entries for this dataset
        conn.execute("DELETE FROM _schema_catalog WHERE dataset_name = ?", [dataset_name])
        for _, row in desc.iterrows():
            conn.execute(
                """
                INSERT INTO _schema_catalog
                    (dataset_name, column_name, data_type, nullable, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                [dataset_name, row["column_name"], row["column_type"], True, None],
            )
    except Exception as exc:
        return f"File loaded as '{dataset_name}' but schema catalog update failed: {exc}"

    row_count = conn.execute(f"SELECT COUNT(*) FROM {dataset_name}").fetchone()[0]
    return (
        f"Successfully loaded '{path}' as table '{dataset_name}'. "
        f"{len(desc)} columns, {row_count} rows registered."
    )


# ── Tool registries (used by LLM nodes) ──────────────────────────────────────────────
ALL_TOOLS = [get_schema, profile_column, run_sql, run_test_query, lookup_semantic, load_file]
ORCHESTRATOR_TOOLS = [get_schema, lookup_semantic]
PROFILER_TOOLS = [get_schema, profile_column]
SQL_WRITER_TOOLS = [get_schema, lookup_semantic, profile_column]
VERIFIER_TOOLS = [run_test_query, profile_column]
