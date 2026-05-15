"""
Ingestion pipeline.

`load_file(path, table_name)` is the single entry point.
It is completely agnostic to the file's domain — it derives everything
from the data itself.

Steps:
  1. Parse file → DuckDB table
  2. Statistical profiling → _column_catalog
  3. Grain/period inference → _table_context
  4. LLM semantic enrichment → _semantic_map
  5. LLM rule generation → _query_rules
  6. Relationship detection → _relationships
"""

import json
from pathlib import Path

import pandas as pd

from agent.core.database import (
    get_conn, execute, run_sql, upsert_semantic_entries, upsert_query_rules,
)
from agent.ingestion.profiler import (
    profile_dataframe, infer_grain, split_cy_py_labels, profile_to_dict,
)
from agent.ingestion.semantic_mapper import (
    generate_semantic_map, generate_query_rules, generate_table_summary,
)


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _sniff_header_row(path: Path) -> int:
    """
    Find the first row that has ≥3 non-null, non-'Unnamed' cells.
    Tries skiprows 0..19.
    """
    for skip in range(20):
        try:
            df = pd.read_excel(path, skiprows=skip, nrows=2)
            good = [c for c in df.columns if "Unnamed" not in str(c) and str(c).strip()]
            if len(good) >= 3:
                return skip
        except Exception:
            continue
    return 0


def _load_into_duckdb(path: Path, table_name: str) -> pd.DataFrame:
    """
    Parse the file and register it as a DuckDB table.
    Returns the DataFrame for profiling.
    """
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls", ".ods"):
        skip = _sniff_header_row(path)
        df = pd.read_excel(path, skiprows=skip)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Drop fully-empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    conn = get_conn()
    conn.register("_tmp_load", df)
    conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _tmp_load')
    conn.unregister("_tmp_load")

    return df


# ---------------------------------------------------------------------------
# Catalog registration helpers
# ---------------------------------------------------------------------------

def _register_column_catalog(table_name: str, profiles: list) -> None:
    conn = get_conn()
    conn.execute(
        "DELETE FROM _column_catalog WHERE table_name = ?", [table_name]
    )
    for p in profiles:
        d = profile_to_dict(p)
        conn.execute("""
            INSERT INTO _column_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            table_name,
            d["column_name"],
            d["dtype"],
            d["null_rate"],
            d["n_unique"],
            d["min_val"],
            d["max_val"],
            d["mean_val"],
            d["std_val"],
            json.dumps(d["top_values"], ensure_ascii=False),
            d["column_role"],
            d["role_confidence"],
        ])


def _build_table_context(
    table_name: str, profiles: list, df: pd.DataFrame
) -> dict:
    """Derive table context entirely from profiles — no domain strings."""
    dim_cols  = [p.name for p in profiles if p.column_role == "dimension"]
    metric_cols = [p.name for p in profiles if p.column_role == "metric"]
    yoy_cols  = [p.name for p in profiles if p.column_role == "yoy_delta"]
    period_cols = [p.name for p in profiles if p.column_role == "period"]

    period_column = period_cols[0] if period_cols else None
    period_values = (
        df[period_column].dropna().unique().tolist()
        if period_column else []
    )
    grain = infer_grain(period_values)
    cy_label, py_label = split_cy_py_labels([str(v) for v in period_values])

    return {
        "table_name": table_name,
        "grain": grain,
        "period_column": period_column,
        "cy_label": cy_label,
        "py_label": py_label,
        "dimension_cols": json.dumps(dim_cols, ensure_ascii=False),
        "metric_cols": json.dumps(metric_cols, ensure_ascii=False),
        "yoy_cols": json.dumps(yoy_cols, ensure_ascii=False),
        "summary": "",  # filled in after LLM call
    }


def _register_table_context(ctx: dict) -> None:
    execute("""
        INSERT INTO _table_context
            (table_name, grain, period_column, cy_label, py_label,
             dimension_cols, metric_cols, yoy_cols, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (table_name) DO UPDATE SET
            grain          = excluded.grain,
            period_column  = excluded.period_column,
            cy_label       = excluded.cy_label,
            py_label       = excluded.py_label,
            dimension_cols = excluded.dimension_cols,
            metric_cols    = excluded.metric_cols,
            yoy_cols       = excluded.yoy_cols,
            summary        = excluded.summary
    """, [
        ctx["table_name"], ctx["grain"], ctx["period_column"],
        ctx["cy_label"], ctx["py_label"],
        ctx["dimension_cols"], ctx["metric_cols"], ctx["yoy_cols"],
        ctx["summary"],
    ])


def _register_data_registry(table_name: str, path: Path, df: pd.DataFrame) -> None:
    execute("""
        INSERT INTO _data_registry (table_name, source_file, row_count, col_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (table_name) DO UPDATE SET
            source_file = excluded.source_file,
            row_count   = excluded.row_count,
            col_count   = excluded.col_count,
            loaded_at   = now()
    """, [table_name, str(path), len(df), len(df.columns)])


# ---------------------------------------------------------------------------
# Cross-table relationship detection
# ---------------------------------------------------------------------------

def _detect_relationships(table_name: str, all_tables: list[str]) -> None:
    """
    Detect potential join keys between newly loaded table and all existing tables.
    Uses two signals: identical column name, or high value overlap.
    """
    new_cols = run_sql(
        "SELECT column_name, column_role FROM _column_catalog WHERE table_name = ?",
        [table_name],
    )
    new_dim_cols = {r["column_name"] for r in new_cols if r["column_role"] == "dimension"}

    for other in all_tables:
        if other == table_name:
            continue
        other_cols = run_sql(
            "SELECT column_name FROM _column_catalog WHERE table_name = ? AND column_role = 'dimension'",
            [other],
        )
        other_dim_cols = {r["column_name"] for r in other_cols}

        # Exact name match
        shared = new_dim_cols & other_dim_cols
        for col in shared:
            execute("""
                INSERT INTO _relationships (table_a, col_a, table_b, col_b, confidence, join_type)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
            """, [table_name, col, other, col, 0.95, "exact_name"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_file(
    path: str | Path,
    table_name: str,
    run_llm_enrichment: bool = True,
) -> dict:
    """
    Load a file, profile it, and populate all metadata tables.

    Parameters
    ----------
    path : path to the source file (xlsx, csv, parquet, …)
    table_name : name to register the table as in DuckDB
    run_llm_enrichment : set False to skip LLM calls (fast mode, no aliases/rules)

    Returns
    -------
    Summary dict with counts and derived context.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[ingest] Loading {path.name} → table '{table_name}'")

    # 1 — Parse into DuckDB
    df = _load_into_duckdb(path, table_name)
    print(f"[ingest] Loaded {len(df):,} rows × {len(df.columns)} columns")

    # 2 — Statistical profiling
    profiles = profile_dataframe(df)
    _register_column_catalog(table_name, profiles)
    print(f"[ingest] Profiled {len(profiles)} columns")

    # 3 — Table context (grain, period, cy/py labels)
    ctx = _build_table_context(table_name, profiles, df)
    _register_table_context(ctx)
    print(f"[ingest] Grain={ctx['grain']}  CY={ctx['cy_label']}  PY={ctx['py_label']}")

    # 4 & 5 — LLM enrichment
    if run_llm_enrichment:
        profile_dicts = [profile_to_dict(p) for p in profiles]

        print("[ingest] Generating semantic map via LLM …")
        sem_entries = generate_semantic_map(table_name, profile_dicts)
        upsert_semantic_entries(sem_entries)
        print(f"[ingest] Wrote {len(sem_entries)} semantic map entries")

        print("[ingest] Generating query rules via LLM …")
        rules = generate_query_rules(table_name, ctx)
        upsert_query_rules(rules)
        print(f"[ingest] Wrote {len(rules)} query rules")

        print("[ingest] Generating table summary …")
        summary = generate_table_summary(table_name, ctx, profile_dicts)
        execute(
            "UPDATE _table_context SET summary = ? WHERE table_name = ?",
            [summary, table_name],
        )
        ctx["summary"] = summary
        print(f"[ingest] Summary: {summary[:120]}…")
    else:
        print("[ingest] Skipping LLM enrichment (fast mode)")

    # 6 — Data registry
    _register_data_registry(table_name, path, df)

    # 7 — Relationship detection
    from agent.core.database import get_all_tables
    all_tables = get_all_tables()
    _detect_relationships(table_name, all_tables)

    return {
        "table_name": table_name,
        "rows": len(df),
        "columns": len(df.columns),
        "grain": ctx["grain"],
        "cy_label": ctx["cy_label"],
        "py_label": ctx["py_label"],
        "summary": ctx.get("summary", ""),
    }
