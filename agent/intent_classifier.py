"""Intent classifier node — runs at START of every graph invocation.

Classifies the user's latest message into one of:
  ingest_file   — user wants to ingest an Excel/CSV file into the KG
  kg_group      — user wants to define/update an entity group (MEMBER_OF)
  kg_derived    — user wants to declare a derived metric (DERIVED_FROM)
  kg_hierarchy  — user wants to define a dimension hierarchy (CHILD_OF/LEVEL_IN)
  kg_review     — user wants to accept/reject a pending MEASURES proposal
  analytics     — normal SQL analytics query (default)

Classification uses a fast regex pass first; falls back to a one-shot LLM
call only when the regex is inconclusive.  The LLM call is intentionally
cheap: it receives only the last user message and returns a single token.

For KG author intents the classifier also extracts a structured payload
from the message so the downstream node can act without a second LLM call.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AnalyticsState

# ---------------------------------------------------------------------------
# Regex heuristics — fast path, no LLM
# ---------------------------------------------------------------------------

_INGEST_RE = re.compile(
    r'\b(ingest|load into (the )?kg|add (to|into) (the )?knowledge.?graph|'
    r'upload|import file|parse (this |the )?file)\b',
    re.IGNORECASE,
)
_GROUP_RE = re.compile(
    r'\b(entity.?group|member.?of|belongs? to|brand.?group|group:)\b',
    re.IGNORECASE,
)
_DERIVED_RE = re.compile(
    r'\b(derived.?metric|derive|ratio|kpi|spend.?per|'
    r'per.?buyer|per.?household|formula|calculated.?metric)\b',
    re.IGNORECASE,
)
_HIERARCHY_RE = re.compile(
    r'\b(hierarchy|level|child.?of|parent|dimension.?tree|subcategor)\b',
    re.IGNORECASE,
)
_REVIEW_RE = re.compile(
    r'\b(accept|reject|approve|decline|review|pending.?proposal|confirm.?concept)\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Payload extractors (best-effort, regex-based)
# ---------------------------------------------------------------------------

def _extract_group_payload(text: str) -> dict[str, Any]:
    """Parse messages like: 'HERISTO group: ANIMONDA, MJAMJAM'"""
    groups = []
    # Pattern: <GROUP_NAME> group: <member1>, <member2>, ...
    for m in re.finditer(
        r'([A-Z][A-Z0-9_]+)\s*(?:group|entity.?group)?[:\s]+([A-Z][A-Z0-9_,\s]+)',
        text, re.IGNORECASE
    ):
        name = m.group(1).strip().upper()
        members_raw = m.group(2)
        members = [
            f'entity::{x.strip().upper()}'
            for x in re.split(r'[,;\s]+', members_raw)
            if x.strip() and x.strip().upper() != name
        ]
        if members:
            groups.append({'name': name, 'members': members})
    return {'groups': groups, 'scoped_to': []}


def _extract_derived_payload(text: str) -> list[dict[str, Any]]:
    """Parse messages like: 'derive SpendPerBuyer = Umsatz / Käuferhaushalte'"""
    decls = []
    for m in re.finditer(
        r'(?:derive|kpi|metric)[:\s]+([\w]+)\s*[=:]\s*([^\n;]+)',
        text, re.IGNORECASE
    ):
        name = m.group(1).strip()
        formula = m.group(2).strip()
        # Extract component ids if mentioned as node ids (metric::...)
        components = re.findall(r'metric::[\w::\s]+', text)
        decls.append({'name': name, 'formula': formula, 'components': components})
    return decls


def _extract_hierarchy_payload(text: str) -> dict[str, Any]:
    """Parse: 'hierarchy CatH: Category > AnimalType > Subcategory'"""
    m = re.search(
        r'(?:hierarchy)[:\s]+([\w]+)[:\s]+([\w\s>|,/]+)',
        text, re.IGNORECASE
    )
    if m:
        root = m.group(1).strip()
        levels = [l.strip() for l in re.split(r'[>|,/]', m.group(2)) if l.strip()]
        return {'root': root, 'levels': levels}
    return {}


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """Classify the user message into exactly one intent token.
Return ONLY the token, nothing else.
Tokens: ingest_file | kg_group | kg_derived | kg_hierarchy | kg_review | analytics

Examples:
- 'ingest cat_mat.xlsx' -> ingest_file
- 'add HERISTO group with ANIMONDA and MJAMJAM' -> kg_group
- 'derive SpendPerBuyer = Umsatz / Käuferhaushalte' -> kg_derived
- 'define hierarchy CatH: Category > Subcategory' -> kg_hierarchy
- 'accept the pending revenue proposal' -> kg_review
- 'show me top brands by Umsatz' -> analytics
"""


def _llm_classify(text: str) -> str:
    """One-shot LLM classification — only called when regex is inconclusive."""
    try:
        from agent.nodes import _get_llm  # reuse same LLM factory
        llm = _get_llm()
        resp = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=text[:500]),  # truncate to keep it cheap
        ])
        token = resp.content.strip().lower().split()[0]
        valid = {'ingest_file', 'kg_group', 'kg_derived', 'kg_hierarchy', 'kg_review', 'analytics'}
        return token if token in valid else 'analytics'
    except Exception:  # noqa: BLE001
        return 'analytics'


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def intent_classifier(state: AnalyticsState) -> dict:
    """Classify intent and extract KG payload from the latest user message."""
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        state.user_query or '',
    )
    text = str(last_human)

    # If ingestion_mode was set externally (e.g. from UI file upload), honour it.
    if state.ingestion_mode and state.kg_ingest_path:
        return {'intent': 'ingest_file'}

    # Fast regex pass
    if _INGEST_RE.search(text):
        # Try to extract a file path from the message
        path_match = re.search(r'[\w./\\-]+\.(?:xlsx|xls|csv)', text, re.IGNORECASE)
        path = path_match.group(0) if path_match else state.kg_ingest_path
        return {
            'intent': 'ingest_file',
            'ingestion_mode': True,
            'kg_ingest_path': path,
        }

    if _GROUP_RE.search(text):
        return {
            'intent': 'kg_group',
            'kg_author_payload': _extract_group_payload(text),
        }

    if _DERIVED_RE.search(text):
        payload = _extract_derived_payload(text)
        if payload:
            return {'intent': 'kg_derived', 'kg_author_payload': {'declarations': payload}}
        # Inconclusive — fall through to LLM

    if _HIERARCHY_RE.search(text):
        payload = _extract_hierarchy_payload(text)
        if payload:
            return {'intent': 'kg_hierarchy', 'kg_author_payload': payload}

    if _REVIEW_RE.search(text):
        return {'intent': 'kg_review', 'kg_author_payload': {'raw': text}}

    # LLM fallback for ambiguous messages
    intent = _llm_classify(text)
    return {'intent': intent}
