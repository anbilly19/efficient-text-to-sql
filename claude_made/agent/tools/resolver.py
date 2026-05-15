"""
Alias resolver.

Resolves natural language terms in a user query to canonical column names
by querying _semantic_map.

Completely retrieval-based — no hardcoded term lists.
"""

import re
import unicodedata

from agent.core.database import run_sql


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).lower().strip()


def resolve_aliases(
    query: str,
    table_name: str,
) -> tuple[str, dict[str, str]]:
    """
    Scan the query for known aliases and return:
      - rewritten query with aliases replaced by canonical column names
      - dict mapping original_term → canonical_col for context injection

    Strategy: longest-first substitution to avoid short aliases shadowing
    longer ones.  Uses exact lowercase match with word-boundary guards.
    """
    entries = run_sql(
        "SELECT alias, canonical_col FROM _semantic_map WHERE table_name = ?",
        [table_name],
    )
    if not entries:
        return query, {}

    # Sort longest alias first
    entries.sort(key=lambda e: len(e["alias"]), reverse=True)

    resolved: dict[str, str] = {}
    rewritten = query

    for entry in entries:
        alias = entry["alias"]
        canonical = entry["canonical_col"]
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, _norm(rewritten), re.IGNORECASE | re.UNICODE):
            resolved[alias] = canonical
            rewritten = re.sub(
                pattern, f'"{canonical}"', rewritten,
                flags=re.IGNORECASE | re.UNICODE,
            )

    return rewritten, resolved


def semantic_search(
    term: str,
    table_name: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Find the most relevant semantic map entries for a term.
    Used for the semantic-lookup fast-path and the Orchestrator's column resolution.
    """
    norm_term = _norm(term)
    if table_name:
        rows = run_sql(
            """
            SELECT alias, canonical_col, description, alias_lang, table_name
            FROM _semantic_map
            WHERE table_name = ?
              AND (LOWER(alias) LIKE ? OR LOWER(canonical_col) LIKE ?)
            LIMIT ?
            """,
            [table_name, f"%{norm_term}%", f"%{norm_term}%", top_k],
        )
    else:
        rows = run_sql(
            """
            SELECT alias, canonical_col, description, alias_lang, table_name
            FROM _semantic_map
            WHERE LOWER(alias) LIKE ? OR LOWER(canonical_col) LIKE ?
            LIMIT ?
            """,
            [f"%{norm_term}%", f"%{norm_term}%", top_k],
        )
    return rows
