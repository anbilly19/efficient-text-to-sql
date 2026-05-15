"""
agent/schema_lookup.py
──────────────────────
Static semantic catalog for the NIQ Haushaltspaneldaten table.

At runtime, this catalog is used ONLY on cold start to seed the DuckDB
_semantic_map table via agent.database._maybe_seed_semantic_map().

All live lookups, alias resolution, and prompt context generation now read
from DuckDB (agent.database.resolve_semantic_term / get_semantic_context)
so that analyst and agent edits persist across restarts.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Column catalog
# Each entry:
#   "exact_column_name": {
#       "en":          human-readable English label,
#       "category":    "dimension" | "metric" | "yoy",
#       "description": one-sentence meaning for the LLM,
#       "aliases":     list[str] – natural-language phrases that map to this column,
#   }
# ---------------------------------------------------------------------------

SCHEMA_CATALOG: dict[str, dict] = {
    "People": {
        "en": "Panel Group",
        "category": "dimension",
        "description": "Panel group identifier. Metadata header rows also live here; filter them out.",
        "aliases": ["panel group", "people group", "audience", "who"],
    },
    "Demographics": {
        "en": "Demographic Segment",
        "category": "dimension",
        "description": "Demographic scope. Always 'Panel gesamt' (total panel) — a fixed-scope indicator.",
        "aliases": ["demographics", "segment", "panel gesamt", "total panel"],
    },
    "Geographies": {
        "en": "Geography / Market",
        "category": "dimension",
        "description": "Geographic scope. Always 'Deutschland gesamt' (all Germany).",
        "aliases": ["geography", "market", "region", "country", "deutschland", "germany"],
    },
    "Periods": {
        "en": "Time Period",
        "category": "dimension",
        "description": (
            "Time window of measurement. Two values: "
            "'Letzte 12 M - 52 W bis 28/12/25' (current year, CY) and "
            "'Letzte 12 M VJ - 52 W bis 29/12/24' (prior year, PY). "
            "Primary time dimension for CY vs. YoY comparisons."
        ),
        "aliases": [
            "period", "time", "year", "CY", "PY", "current year", "prior year",
            "vorjahr", "VJ", "date range", "when", "12 months", "letzte 12",
        ],
    },
    "Products": {
        "en": "Product / SKU",
        "category": "dimension",
        "description": "Product or SKU name within the Saturn Petcare / petfood category. 648 unique values.",
        "aliases": [
            "product", "SKU", "item", "brand", "article", "produkt", "artikel",
            "petfood", "cat food", "dog food", "what product", "which product",
        ],
    },
    "Retailers": {
        "en": "Retailer / Channel",
        "category": "dimension",
        "description": "Retail channel or store (9 unique values: e.g. Rewe, Edeka, Amazon, Zooplus).",
        "aliases": [
            "retailer", "store", "channel", "shop", "where bought",
            "händler", "kanal", "online", "offline",
        ],
    },
    "Penetration (%)": {
        "en": "Penetration Rate % (Current Year)",
        "category": "metric",
        "description": (
            "% of all German households that bought this product/retailer in the current 12-month period. "
            "Range: ~0–12.9%. Core reach KPI. Also referred to as Käuferreichweite in German NIQ reports."
        ),
        "aliases": [
            "penetration", "reach", "how many households bought", "buyer rate",
            "market penetration", "household share", "penetrationsrate", "% haushalte",
            "käuferreichweite", "kaeuferreichweite", "buyer reach", "reichweite",
        ],
    },
    "Penetration (%) VJ": {
        "en": "Penetration Rate % (Prior Year)",
        "category": "yoy",
        "description": "Penetration rate for the prior year. Baseline for YoY reach comparison.",
        "aliases": [
            "penetration VJ", "prior year penetration", "last year penetration",
            "vorjahr penetration", "käuferreichweite VJ", "reichweite VJ",
        ],
    },
    "Penetration (%) vs. VJ (% Ver.)": {
        "en": "Penetration Change vs. Prior Year (%)",
        "category": "yoy",
        "description": (
            "% change in penetration vs. prior year. Range: -100% to +11,318%. "
            "Extreme positives = new launches. This column is PRECOMPUTED by NIQ — "
            "select it directly with AVG(); never recompute via LAG() or CY minus PY."
        ),
        "aliases": [
            "penetration growth", "penetration change", "YoY penetration",
            "veränderung penetration", "reach growth", "gained buyers",
            # full-phrase German variants (all word orders)
            "veränderung zum vorjahr penetration",
            "veränderung der penetration",
            "veränderung zum vorjahr der penetration",
            "penetration veränderung zum vorjahr",
            "penetration änderung vorjahr",
            "penetration yoy",
            "yoy penetration change",
            # bare YoY trigger phrases — these alone should resolve to this column
            # when the context is clearly about penetration
            "reichweite wachstum",
            "käuferreichweite veränderung",
            "käuferreichweite yoy",
            "reichweite veränderung",
            "reichweite änderung",
        ],
    },
    "Käuferhaushalte": {
        "en": "Buying Households (Current Year)",
        "category": "metric",
        "description": "Absolute number of German households that bought this product at this retailer (CY). Range: 61–5.4M.",
        "aliases": [
            "buying households", "buyers", "households",
            "how many buyers", "buyer count", "anzahl käufer", "kaufer",
        ],
    },
    "Käuferhaushalte VJ": {
        "en": "Buying Households (Prior Year)",
        "category": "yoy",
        "description": "Number of buying households in the prior year. Baseline for YoY buyer count comparison.",
        "aliases": ["buyers last year", "prior year buyers", "vorjahr käufer", "käufer VJ"],
    },
    "Käuferhaushalte vs. VJ (% Ver.)": {
        "en": "Buying Households Change vs. Prior Year (%)",
        "category": "yoy",
        "description": "% change in absolute buying household count vs. prior year.",
        "aliases": ["buyer growth", "household growth", "more buyers", "käufer wachstum", "buyer change YoY"],
    },
    "Einkaufsakte pro Käuferhaushalt": {
        "en": "Purchase Frequency per Buying Household (CY)",
        "category": "metric",
        "description": (
            "Avg number of purchase occasions per buying household in the current period. "
            "Range: 1–24. High value = repeat buying / loyalty."
        ),
        "aliases": [
            "purchase frequency", "buy frequency", "occasions", "trips",
            "kauffrequenz", "frequenz", "how often", "repeat purchase", "loyalty",
            "einkaufsakte", "frequency",
        ],
    },
    "Einkaufsakte pro Käuferhaushalt VJ": {
        "en": "Purchase Frequency per Buying Household (Prior Year)",
        "category": "yoy",
        "description": "Purchase frequency per buying household in the prior year.",
        "aliases": ["frequency VJ", "prior year frequency", "frequenz VJ", "vorjahr frequenz"],
    },
    "Einkaufsakte pro Käuferhaushalt vs. VJ (% Ver.)": {
        "en": "Purchase Frequency Change vs. Prior Year (%)",
        "category": "yoy",
        "description": "% change in purchase frequency vs. prior year. Mean: -3.4% (slight overall decline).",
        "aliases": [
            "frequency change", "frequency growth", "buy more often", "frequenz veränderung",
            "veränderung frequenz vorjahr", "frequenz yoy",
        ],
    },
    "Anzahl Einkaufsakte": {
        "en": "Total Purchase Occasions (Current Year)",
        "category": "metric",
        "description": (
            "Total purchase trips for this product/retailer in CY. "
            "= Käuferhaushalte × Einkaufsakte per household. Range: 61–50M."
        ),
        "aliases": [
            "total occasions", "total trips", "total purchases", "volume",
            "anzahl", "einkaufsakte gesamt", "how many purchases", "purchase count",
            "transaction count",
        ],
    },
    "Anzahl Einkaufsakte VJ": {
        "en": "Total Purchase Occasions (Prior Year)",
        "category": "yoy",
        "description": "Total purchase occasions in the prior year.",
        "aliases": ["occasions VJ", "prior year volume", "anzahl VJ", "last year purchases"],
    },
    "Anzahl Einkaufsakte vs. VJ (% Ver.)": {
        "en": "Total Purchase Occasions Change vs. Prior Year (%)",
        "category": "yoy",
        "description": "% change in total purchase volume vs. prior year. Mean: +37%.",
        "aliases": [
            "volume growth", "occasion growth", "anzahl wachstum", "trip growth YoY", "volume change",
            "veränderung anzahl vorjahr", "anzahl yoy",
        ],
    },
    "Ausgaben pro Käuferhaushalt": {
        "en": "Spend per Buying Household € (Current Year)",
        "category": "metric",
        "description": (
            "Avg annual spend (€) per buying household in CY. Range: €0.01–€823. Key value/wallet metric. "
            "This is a PRE-AGGREGATED NIQ metric — use AVG() when grouping, never SUM()."
        ),
        "aliases": [
            "spend per household", "average spend", "ausgaben", "wallet",
            "how much spent", "consumer value", "expenditure", "ausgaben haushalt",
            "spend", "euro per buyer", "€ per buyer",
            "ausgaben je käufer", "ausgaben pro käufer", "spend per buyer",
        ],
    },
    "Ausgaben pro Käuferhaushalt VJ": {
        "en": "Spend per Buying Household € (Prior Year)",
        "category": "yoy",
        "description": "Avg annual spend per buying household in the prior year.",
        "aliases": ["spend VJ", "prior year spend", "ausgaben VJ", "last year spend"],
    },
    "Ausgaben pro Käuferhaushalt vs. VJ (% Ver.)": {
        "en": "Spend per Buying Household Change vs. Prior Year (%)",
        "category": "yoy",
        "description": "% change in spend per buying household vs. prior year.",
        "aliases": [
            "spend growth", "spend change YoY", "ausgaben wachstum",
            "veränderung ausgaben vorjahr", "ausgaben yoy",
        ],
    },
    "Ausgaben pro Einkaufsakt": {
        "en": "Spend per Purchase Occasion € (Current Year)",
        "category": "metric",
        "description": "Avg basket value (€) per individual purchase trip in CY. Range: €0.01–€227.88. Mean: ~€10.",
        "aliases": [
            "basket size", "spend per trip", "spend per occasion", "price per purchase",
            "basket value", "korb", "ausgaben akt", "how much per visit", "ticket size",
            "average basket",
        ],
    },
    "Ausgaben pro Einkaufsakt VJ": {
        "en": "Spend per Purchase Occasion € (Prior Year)",
        "category": "yoy",
        "description": "Avg basket value per purchase occasion in the prior year.",
        "aliases": ["basket VJ", "prior year basket", "ausgaben akt VJ", "spend per trip last year"],
    },
}
