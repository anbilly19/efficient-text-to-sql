-- Migration 001: multi-table metadata bootstrap
-- Run once against a fresh DB (idempotent — uses IF NOT EXISTS / ON CONFLICT).

-- _data_registry, _column_catalog, _relationships, _table_context and
-- _semantic_map are created at connection time by database._ensure_metadata_tables().
-- This migration only needs to seed the static join relationships and
-- table-context descriptions for the two derived seed tables.

-- ── seed table context ────────────────────────────────────────────────────

INSERT INTO _table_context (dataset_name, summary, grain, tags)
VALUES
  (
    'sales_rep_targets',
    'Per-sales-rep historical revenue and quota targets, derived from sales1000.',
    'one row per sales_rep + region',
    '["targets", "quota", "sales_rep", "region"]'
  ),
  (
    'product_metrics',
    'Per-product aggregated sales volume and revenue, derived from sales1000.',
    'one row per product_name',
    '["product", "volume", "revenue"]'
  )
ON CONFLICT (dataset_name) DO UPDATE
  SET summary    = excluded.summary,
      grain      = excluded.grain,
      tags       = excluded.tags,
      updated_at = current_timestamp;

-- ── seed explicit join relationships ─────────────────────────────────────
-- sales1000 → sales_rep_targets on sales_rep (many-to-one)
INSERT INTO _relationships
  (relationship_id, left_table, left_column, right_table, right_column, cardinality, description)
VALUES
  (
    nextval('_rel_seq'),
    'sales1000', 'Sales Rep',
    'sales_rep_targets', 'sales_rep',
    'many-to-one',
    'Each sales1000 row belongs to one sales rep target row'
  ),
  (
    nextval('_rel_seq'),
    'sales1000', 'Region',
    'sales_rep_targets', 'region',
    'many-to-one',
    'Each sales1000 row belongs to one region in sales_rep_targets'
  ),
  (
    nextval('_rel_seq'),
    'sales1000', 'Product Name',
    'product_metrics', 'product_name',
    'many-to-one',
    'Each sales1000 row joins to one product_metrics row'
  )
ON CONFLICT (left_table, left_column, right_table, right_column) DO NOTHING;

-- Mark join keys in _column_catalog (runs after index_table_schema() has populated it)
UPDATE _column_catalog SET is_join_key = TRUE
WHERE (dataset_name = 'sales1000'          AND column_name IN ('Sales Rep', 'Region', 'Product Name'))
   OR (dataset_name = 'sales_rep_targets'  AND column_name IN ('sales_rep', 'region'))
   OR (dataset_name = 'product_metrics'    AND column_name = 'product_name');
