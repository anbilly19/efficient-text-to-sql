# This file has been superseded.
# 
# - To generate data/product_metrics.xlsx (standalone, no DB needed):
#       python scripts/generate_product_metrics.py
#
# - sales_rep_targets is derived from sales1000 inside the agent at load time
#   via db/migrations/001_multi_table.sql — no manual seeding required.
#
# This stub is kept to avoid import errors in any scripts that reference it.
# It will be removed in a future cleanup commit.

raise ImportError(
    "seed_multi_table.py has been removed. "
    "Use scripts/generate_product_metrics.py for product_metrics.xlsx generation."
)
