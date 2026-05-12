-- Seed: sales_rep_targets
-- Materialized from sales1000 via GROUP BY sales_rep + region.
-- quota = historical_revenue * 1.15  (a 15 % uplift target).
--
-- Run AFTER sales1000 has been loaded into DuckDB.

CREATE OR REPLACE TABLE sales_rep_targets AS
SELECT
    "Sales Rep"                           AS sales_rep,
    "Region"                              AS region,
    SUM("Revenue")                        AS historical_revenue,
    ROUND(SUM("Revenue") * 1.15, 2)       AS quota,
    COUNT(*)                              AS total_orders,
    ROUND(AVG("Revenue"), 2)              AS avg_order_revenue,
    MIN("Order Date")                     AS first_order_date,
    MAX("Order Date")                     AS last_order_date
FROM sales1000
GROUP BY "Sales Rep", "Region"
ORDER BY historical_revenue DESC;

-- Register table context
INSERT INTO _table_context (dataset_name, summary, grain, tags)
VALUES (
    'sales_rep_targets',
    'Per-sales-rep historical revenue and quota targets, derived from sales1000.',
    'one row per sales_rep + region',
    '["targets", "quota", "sales_rep", "region"]'
)
ON CONFLICT (dataset_name) DO UPDATE
  SET summary    = excluded.summary,
      grain      = excluded.grain,
      tags       = excluded.tags,
      updated_at = current_timestamp;
