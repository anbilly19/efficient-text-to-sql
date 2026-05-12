# REMOVED: this script materialised sales_rep_targets and product_metrics by
# running GROUP BY against sales1000 at setup time, which is the wrong design.
#
# Additional tables must be independent data sources uploaded via the agent.
# To generate independent test Excel files run:
#
#   python scripts/generate_test_data.py
#
# Then upload them:
#   load file at data/sales_rep_targets.xlsx as sales_rep_targets
#   load file at data/product_metrics.xlsx as product_metrics
