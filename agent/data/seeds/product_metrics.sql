-- Seed: product_metrics
-- Materialized from sales1000 via GROUP BY product_name.
-- Captures aggregate volume and revenue per product.
--
-- Run AFTER sales1000 has been loaded into DuckDB.

CREATE OR REPLACE TABLE product_metrics AS
SELECT
    "Product Name"                        AS product_name,
    "Category"                            AS category,
    COUNT(*)                              AS total_orders,
    SUM("Units Sold")                     AS total_units_sold,
    ROUND(SUM("Revenue"), 2)              AS total_revenue,
    ROUND(AVG("Revenue"), 2)              AS avg_order_revenue,
    ROUND(AVG("Units Sold"), 2)           AS avg_units_per_order,
    MIN("Order Date")                     AS first_seen,
    MAX("Order Date")                     AS last_seen
FROM sales1000
GROUP BY "Product Name", "Category"
ORDER BY total_revenue DESC;

-- Register table context
INSERT INTO _table_context (dataset_name, summary, grain, tags)
VALUES (
    'product_metrics',
    'Per-product aggregated sales volume and revenue, derived from sales1000.',
    'one row per product_name',
    '["product", "volume", "revenue", "category"]'
)
ON CONFLICT (dataset_name) DO UPDATE
  SET summary    = excluded.summary,
      grain      = excluded.grain,
      tags       = excluded.tags,
      updated_at = current_timestamp;
