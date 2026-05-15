"""
LLM-assisted semantic enrichment.

Takes raw column profiles and asks the LLM to generate:
  - Natural language aliases (all languages found in the data)
  - Plain-English descriptions
  - Table-level query rules derived from the data structure

No domain strings are hardcoded here.  Everything is derived from the
actual column names, dtypes, and sample values passed in at runtime.
"""

import json
import re
from typing import Any

import anthropic

_client = anthropic.Anthropic()
_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_semantic_prompt(table_name: str, profiles: list[dict]) -> str:
    cols_block = json.dumps(profiles, ensure_ascii=False, indent=2)
    return f"""You are a data dictionary specialist. You will be given column profiles
from a dataset called "{table_name}". Your job is to produce semantic metadata
that will help a natural-language SQL agent understand what each column means
and how users might refer to it.

## Column profiles
{cols_block}

## Your task
For EACH column, return a JSON object with these fields:
- "column_name": exact column name as given
- "description": 1-2 sentence plain English explanation of what this column measures
- "aliases": list of 4-8 natural language terms a business analyst might use
  (include abbreviations, synonyms, and translations if the column name is
   in a non-English language — detect the language automatically)
- "alias_lang": for each alias specify the language code ("en", "de", "fr", "abbrev", etc.)
  as a parallel list to "aliases"

Return a JSON array of these objects — one per column.
Return ONLY the JSON array, no markdown, no preamble.

Rules:
- Aliases must be lowercase
- Include at minimum 2 English aliases per column even if the original is in another language
- For yoy_delta columns, aliases should include "year over year", "vs prior year", "change vs last year"
- For period columns, aliases should include "time period", "date range", "reporting period"
- For prior_period columns, aliases should include "prior year", "last year", "previous period"
- Be specific: "penetration rate" not just "rate"
"""


def _build_rules_prompt(table_name: str, context: dict) -> str:
    return f"""You are a SQL correctness specialist. Given the structural metadata below
for a dataset called "{table_name}", generate SQL query rules that should ALWAYS be
applied when querying this dataset to avoid incorrect results.

## Table metadata
- Grain: {context.get('grain')}
- Period column: {context.get('period_column')}
- Current-year label: {context.get('cy_label')}
- Prior-year label: {context.get('py_label')}
- Dimension columns: {context.get('dimension_cols')}
- Metric columns: {context.get('metric_cols')}
- YoY delta columns: {context.get('yoy_cols')}

## Your task
Generate SQL query rules. Each rule must fire when certain SQL patterns are detected
and inject a corrective SQL fragment.

Return a JSON array where each element has:
- "rule_id": integer starting at 1
- "condition_tag": short snake_case label (e.g. "uses_yoy_col", "group_query", "missing_period_filter")
- "condition_sql": a regex pattern that, when matched against the generated SQL, triggers this rule
- "inject_fragment": the SQL fragment to add or check (use {{cy_label}} as placeholder for the CY period value)
- "inject_position": one of "WHERE" | "GROUP_BY" | "HAVING" | "advisory"
- "explanation": why this rule exists, in plain English

Return ONLY the JSON array. No markdown.

Generate only rules that are structurally necessary given the metadata above.
Do NOT generate rules based on assumptions about the domain — only from the data structure.
"""


def _build_summary_prompt(table_name: str, context: dict, profiles: list[dict]) -> str:
    dim_samples = {
        p["column_name"]: p["top_values"]
        for p in profiles
        if p["column_role"] == "dimension"
    }
    return f"""Given this dataset metadata, write a 2-3 sentence plain English summary
of what this dataset contains and what analytical questions it can answer.

Dataset name: {table_name}
Grain: {context.get('grain')}
Dimensions: {list(context.get('dimension_cols', []))}
Key metrics: {list(context.get('metric_cols', []))[:6]}
Sample dimension values: {json.dumps(dim_samples, ensure_ascii=False)}

Return only the summary paragraph, no headers or preamble.
"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_tokens: int = 4000) -> str:
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_json_response(raw: str) -> Any:
    """Strip any accidental markdown fences and parse."""
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_semantic_map(
    table_name: str,
    profiles: list[dict],
    batch_size: int = 20,
) -> list[dict]:
    """
    Generate semantic map entries for every column in the table.
    Batches LLM calls to stay within token limits.
    Returns list of _semantic_map rows ready for upsert.
    """
    entries = []
    for i in range(0, len(profiles), batch_size):
        batch = profiles[i : i + batch_size]
        prompt = _build_semantic_prompt(table_name, batch)
        raw = _call_llm(prompt)
        try:
            col_metas = _parse_json_response(raw)
        except json.JSONDecodeError:
            continue  # log and skip bad batch in production

        for meta in col_metas:
            col = meta.get("column_name", "")
            aliases = meta.get("aliases", [])
            langs = meta.get("alias_lang", ["en"] * len(aliases))
            desc = meta.get("description", "")

            # Always include the raw column name as an alias too
            entries.append({
                "table_name": table_name,
                "alias": col.lower(),
                "canonical_col": col,
                "description": desc,
                "alias_lang": "raw",
            })
            for alias, lang in zip(aliases, langs):
                if alias:
                    entries.append({
                        "table_name": table_name,
                        "alias": alias.lower().strip(),
                        "canonical_col": col,
                        "description": desc,
                        "alias_lang": lang,
                    })

    return entries


def generate_query_rules(table_name: str, context: dict) -> list[dict]:
    """
    Generate data-derived query rules for the Verifier.
    Returns list of _query_rules rows.
    """
    if not context.get("cy_label"):
        return []  # Can't generate meaningful rules without period context

    prompt = _build_rules_prompt(table_name, context)
    raw = _call_llm(prompt, max_tokens=2000)
    try:
        rules = _parse_json_response(raw)
    except json.JSONDecodeError:
        return []

    for r in rules:
        r["table_name"] = table_name
        # Substitute the actual CY label into inject fragments
        if "inject_fragment" in r and context.get("cy_label"):
            r["inject_fragment"] = r["inject_fragment"].replace(
                "{cy_label}", context["cy_label"]
            )
    return rules


def generate_table_summary(
    table_name: str,
    context: dict,
    profiles: list[dict],
) -> str:
    prompt = _build_summary_prompt(table_name, context, profiles)
    return _call_llm(prompt, max_tokens=300)
