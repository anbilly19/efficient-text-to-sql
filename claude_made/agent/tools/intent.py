"""
Intent classifier.

Classifies user queries into routing intents by asking the LLM,
seeded with actual metadata from the loaded tables.

No hardcoded regex fast-paths for domain-specific intents.
The only deterministic shortcuts are for truly structural operations
(file loading, schema listing) that are format-agnostic.
"""

import re
from dataclasses import dataclass
from enum import Enum

import anthropic

from agent.core.database import get_all_tables, run_sql

_client = anthropic.Anthropic()
_MODEL = "claude-sonnet-4-20250514"


class Intent(str, Enum):
    LOAD_FILE       = "load_file"        # "load <path> as <name>"
    SCHEMA_LOOKUP   = "schema_lookup"    # "what columns does X have"
    DISTINCT_VALUES = "distinct_values"  # "what values are in column X of table Y"
    GRAIN_QUERY     = "grain_query"      # "what is the time grain of X"
    META_QUERY      = "meta_query"       # "what tables are loaded"
    SEMANTIC_LOOKUP = "semantic_lookup"  # "what does term X map to"
    ANALYTICS       = "analytics"        # actual data question → LLM pipeline
    ADVERSARIAL     = "adversarial"      # references concept with no matching column
    CLARIFY         = "clarify"          # too ambiguous to classify


@dataclass
class ClassifiedIntent:
    intent: Intent
    table_name: str | None
    extra: dict  # e.g. {"column": "Periods"} for distinct_values


# ---------------------------------------------------------------------------
# Deterministic structural shortcuts — truly format-agnostic
# ---------------------------------------------------------------------------

_LOAD_RE = re.compile(
    r"^load\s+\S+\s+as\s+\w+", re.IGNORECASE
)
_SCHEMA_RE = re.compile(
    r"\b(what columns|show schema|describe table|show columns|list columns"
    r"|welche spalten|spalten von|struktur von)\b",
    re.IGNORECASE | re.UNICODE,
)
_META_RE = re.compile(
    r"\b(what tables|which tables|tables loaded|available tables"
    r"|welche tabellen|geladene tabellen)\b",
    re.IGNORECASE | re.UNICODE,
)


def _try_deterministic(query: str, all_tables: list[str]) -> ClassifiedIntent | None:
    if _LOAD_RE.match(query.strip()):
        return ClassifiedIntent(Intent.LOAD_FILE, None, {})
    if _SCHEMA_RE.search(query):
        table = _extract_table_mention(query, all_tables)
        return ClassifiedIntent(Intent.SCHEMA_LOOKUP, table, {})
    if _META_RE.search(query):
        return ClassifiedIntent(Intent.META_QUERY, None, {})
    return None


def _extract_table_mention(query: str, all_tables: list[str]) -> str | None:
    q_lower = query.lower()
    for t in all_tables:
        if t.lower() in q_lower:
            return t
    return all_tables[0] if all_tables else None


# ---------------------------------------------------------------------------
# LLM-based classification (for everything else)
# ---------------------------------------------------------------------------

def _build_classify_prompt(query: str, all_tables: list[str]) -> str:
    # Fetch lightweight metadata to give the LLM context about what's loaded
    table_summaries = []
    for t in all_tables:
        ctx = run_sql("SELECT grain, cy_label, py_label, summary FROM _table_context WHERE table_name = ?", [t])
        cols = run_sql("SELECT column_name, column_role FROM _column_catalog WHERE table_name = ? LIMIT 30", [t])
        col_list = [(c["column_name"], c["column_role"]) for c in cols]
        table_summaries.append({
            "table": t,
            "context": ctx[0] if ctx else {},
            "columns_sample": col_list,
        })

    import json
    return f"""Classify the following user query into one of these intents:

- "distinct_values": user wants to know what values exist in a specific column
- "grain_query": user asks about the time grain or granularity of a table
- "semantic_lookup": user asks what a term or column name means
- "analytics": user wants actual data analysis, aggregation, or SQL results
- "adversarial": user references a concept that clearly doesn't exist in any loaded table
- "clarify": query is too vague to classify

## Loaded tables
{json.dumps(table_summaries, ensure_ascii=False, indent=2)}

## User query
{query}

Return JSON:
{{
  "intent": "<one of the intent names above>",
  "table_name": "<most relevant table name or null>",
  "extra": {{}}  // for distinct_values: {{"column": "<column_name>"}}
                 // for semantic_lookup: {{"term": "<term>"}}
}}

Return only JSON.
"""


def classify(query: str) -> ClassifiedIntent:
    all_tables = get_all_tables()

    # Try cheap deterministic shortcuts first
    det = _try_deterministic(query, all_tables)
    if det:
        return det

    if not all_tables:
        return ClassifiedIntent(Intent.META_QUERY, None, {})

    prompt = _build_classify_prompt(query, all_tables)
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    import json
    import re as _re
    cleaned = _re.sub(r"^```[a-z]*\n?", "", raw, flags=_re.MULTILINE)
    cleaned = _re.sub(r"\n?```$", "", cleaned.strip(), flags=_re.MULTILINE)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return ClassifiedIntent(Intent.ANALYTICS, all_tables[0] if all_tables else None, {})

    intent_str = parsed.get("intent", "analytics")
    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.ANALYTICS

    return ClassifiedIntent(
        intent=intent,
        table_name=parsed.get("table_name"),
        extra=parsed.get("extra", {}),
    )
