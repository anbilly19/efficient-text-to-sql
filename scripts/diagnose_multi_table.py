#!/usr/bin/env python
"""
scripts/diagnose_multi_table.py
--------------------------------
Connects to the real DuckDB (same path the agent uses) and runs the
five multi-table queries that surfaced issues, printing annotated results.

Usage:
    python scripts/diagnose_multi_table.py

    # Override DB path:
    DUCKDB_PATH=/path/to/db.duckdb python scripts/diagnose_multi_table.py
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

# Force UTF-8 output on Windows so box-drawing chars don't crash cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb

DB_PATH = os.getenv(
    "DUCKDB_PATH",
    str(ROOT / ".local" / "duckdb" / "efficient-text-to-sql.duckdb"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "\n" + "=" * 72 + "\n"


def run(conn: duckdb.DuckDBPyConnection, label: str, sql: str) -> None:
    print("\n" + "-" * 60)
    print(f"[{label}]")
    print(textwrap.indent(sql.strip(), "  "))
    print()
    try:
        rows = conn.execute(sql).fetchall()
        desc = conn.description
        if not rows:
            print("  [!] No rows returned.")
            return
        headers = [d[0] for d in desc]
        col_w = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  " + "  ".join("-" * w for w in col_w))
        for row in rows[:20]:
            print(fmt.format(*[str(v) for v in row]))
        if len(rows) > 20:
            print(f"  ... ({len(rows)} rows total, showing first 20)")
        print(f"  -> {len(rows)} row(s)")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def section(title: str) -> None:
    print(SEP + f"  {title}" + SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Connecting to: {DB_PATH}")
    if not Path(DB_PATH).exists():
        print("ERROR: DB file not found. Run the agent at least once to create it.")
        sys.exit(1)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # --- 1. Inventory -------------------------------------------------------
    section("1. LOADED TABLES")
    run(conn, "data_registry",
        "SELECT dataset_name, row_count, column_count, source_file "
        "FROM _data_registry ORDER BY ingested_at")

    # --- 2. Column schemas --------------------------------------------------
    section("2. COLUMN SCHEMAS")
    for tbl in ("sales1000", "sales_rep_targets", "product_metrics"):
        run(conn, f"columns: {tbl}",
            f"SELECT column_name, column_type, is_join_key "
            f"FROM _column_catalog WHERE dataset_name = '{tbl}' ORDER BY column_name")

    # --- 3. Registered relationships ----------------------------------------
    section("3. REGISTERED RELATIONSHIPS")
    run(conn, "_relationships",
        "SELECT left_table, left_column, right_table, right_column, cardinality "
        "FROM _relationships ORDER BY relationship_id")

    # --- 4. Join key overlap -------------------------------------------------
    section("4. JOIN KEY OVERLAP CHECK")
    run(conn, "distinct regions in sales1000",
        'SELECT DISTINCT "region" FROM sales1000 ORDER BY 1')
    run(conn, "distinct region in sales_rep_targets",
        'SELECT DISTINCT "region" FROM sales_rep_targets ORDER BY 1')
    run(conn, "rep names in sales_rep_targets",
        'SELECT DISTINCT "sales_rep" FROM sales_rep_targets ORDER BY 1')
    run(conn, "rep names in sales1000 (sample)",
        'SELECT DISTINCT "sales_rep" FROM sales1000 ORDER BY 1 LIMIT 15')
    run(conn, "join coverage (matching rep+region pairs)",
        """
        SELECT COUNT(*) AS matched_pairs
        FROM sales_rep_targets t
        INNER JOIN (
            SELECT DISTINCT "sales_rep", "region" FROM sales1000
        ) s ON t."sales_rep" = s."sales_rep" AND t."region" = s."region"
        """)

    # --- 5. product_metrics column names ------------------------------------
    section("5. PRODUCT_METRICS ACTUAL COLUMN NAMES")
    run(conn, "product_metrics columns",
        "SELECT column_name, column_type FROM _column_catalog "
        "WHERE dataset_name = 'product_metrics' ORDER BY column_name")
    run(conn, "product_metrics sample rows",
        "SELECT * FROM product_metrics LIMIT 5")

    # --- 6. Diagnostic queries ----------------------------------------------
    section("6. DIAGNOSTIC QUERIES")

    run(conn, "Q1a: revenue vs quota per rep+region",
        """
        SELECT s."sales_rep", s."region",
               SUM(s."total_revenue")  AS actual_revenue,
               t."quota_usd"           AS quota,
               SUM(s."total_revenue") - t."quota_usd" AS gap
        FROM sales1000 s
        JOIN sales_rep_targets t
          ON s."sales_rep" = t."sales_rep" AND s."region" = t."region"
        GROUP BY s."sales_rep", s."region", t."quota_usd"
        ORDER BY gap
        """)

    run(conn, "Q1b: below-quota reps",
        """
        SELECT s."sales_rep", s."region",
               SUM(s."total_revenue")  AS actual_revenue,
               t."quota_usd"           AS quota,
               SUM(s."total_revenue") - t."quota_usd" AS gap
        FROM sales1000 s
        JOIN sales_rep_targets t
          ON s."sales_rep" = t."sales_rep" AND s."region" = t."region"
        GROUP BY s."sales_rep", s."region", t."quota_usd"
        HAVING SUM(s."total_revenue") < t."quota_usd"
        ORDER BY gap
        """)

    run(conn, "Q2: quota attainment ranking",
        """
        SELECT t."region", t."sales_rep",
               ROUND(SUM(s."total_revenue") / t."quota_usd" * 100, 1) AS attainment_pct
        FROM sales1000 s
        JOIN sales_rep_targets t
          ON s."sales_rep" = t."sales_rep" AND s."region" = t."region"
        GROUP BY t."region", t."sales_rep", t."quota_usd"
        ORDER BY t."region", attainment_pct DESC
        """)

    run(conn, "Q3: total quota gap",
        """
        SELECT
            SUM(t."quota_usd")              AS total_quota,
            SUM(s.actual_revenue)           AS total_actual,
            SUM(t."quota_usd") - SUM(s.actual_revenue) AS total_gap
        FROM sales_rep_targets t
        LEFT JOIN (
            SELECT "sales_rep", "region", SUM("total_revenue") AS actual_revenue
            FROM sales1000
            GROUP BY "sales_rep", "region"
        ) s ON t."sales_rep" = s."sales_rep" AND t."region" = s."region"
        """)

    run(conn, "Q4: top-5 products by revenue",
        """
        SELECT "product", SUM("total_revenue") AS total_rev
        FROM sales1000
        GROUP BY "product"
        ORDER BY total_rev DESC
        LIMIT 5
        """)

    run(conn, "Q4: below-quota reps + top-5 products",
        """
        WITH top5 AS (
            SELECT "product"
            FROM sales1000
            GROUP BY "product"
            ORDER BY SUM("total_revenue") DESC
            LIMIT 5
        ),
        rep_revenue AS (
            SELECT "sales_rep", "region", SUM("total_revenue") AS actual
            FROM sales1000
            GROUP BY "sales_rep", "region"
        ),
        below_quota AS (
            SELECT t."sales_rep", t."region",
                   t."quota_usd" - r.actual AS quota_gap
            FROM sales_rep_targets t
            JOIN rep_revenue r
              ON t."sales_rep" = r."sales_rep" AND t."region" = r."region"
            WHERE r.actual < t."quota_usd"
        ),
        rep_top_product AS (
            SELECT s."sales_rep", s."region", s."product",
                   SUM(s."total_revenue") AS product_rev,
                   ROW_NUMBER() OVER (
                       PARTITION BY s."sales_rep", s."region"
                       ORDER BY SUM(s."total_revenue") DESC
                   ) AS rn
            FROM sales1000 s
            JOIN top5 t5 ON s."product" = t5."product"
            GROUP BY s."sales_rep", s."region", s."product"
        )
        SELECT bq."sales_rep", bq."region", bq.quota_gap,
               rtp."product" AS top_product
        FROM below_quota bq
        JOIN rep_top_product rtp
          ON bq."sales_rep" = rtp."sales_rep"
         AND bq."region"    = rtp."region"
         AND rtp.rn = 1
        ORDER BY bq.quota_gap DESC
        """)

    run(conn, "Q5: revenue share of quota by rep+category",
        """
        WITH rep_quota AS (
            SELECT "sales_rep", "region", "quota_usd"
            FROM sales_rep_targets
        )
        SELECT s."sales_rep",
               s."category",
               SUM(s."total_revenue")                              AS category_revenue,
               ROUND(
                   SUM(s."total_revenue") / MAX(q."quota_usd") * 100,
               2)                                                  AS quota_share_pct
        FROM sales1000 s
        JOIN rep_quota q
          ON s."sales_rep" = q."sales_rep" AND s."region" = q."region"
        GROUP BY s."sales_rep", s."category"
        ORDER BY s."sales_rep", quota_share_pct DESC
        """)

    print(SEP + "  Done." + SEP)
    conn.close()


if __name__ == "__main__":
    main()
